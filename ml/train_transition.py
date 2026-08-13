# ml/train_transition.py
import json

import numpy as np
import torch
import torch.nn as nn

from lib.onnx_export import export_model

FEATURES = ["energy", "rising", "bpmDelta"]  # hasRealStructure dropped — it's Task 9's gate on whether the AI path runs, not a model input (see Task 6's "Design correction")
TARGETS = ["transitionMs", "duckDb", "fxIntensity"]


class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, len(TARGETS)),
        )

    def forward(self, x):
        return self.net(x)


def main():
    with open("data/transition_dataset.json") as f:
        rows = json.load(f)
    if len(rows) < 20:
        raise SystemExit(f"Only {len(rows)} rows — too few to train on. Check Task 6/7 output.")

    x = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[r[k] for k in TARGETS] for r in rows], dtype=np.float32)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std == 0] = 1.0

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    n_val = max(1, int(0.15 * len(x)))
    idx = np.random.default_rng(0).permutation(len(x))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.tensor(x_norm[train_idx])
    y_train = torch.tensor(y_norm[train_idx])
    x_val = torch.tensor(x_norm[val_idx])
    y_val = torch.tensor(y_norm[val_idx])

    model = TransitionNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience, patience_left = 15, 15
    for epoch in range(300):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save(model.state_dict(), "export/transition_best.pt")  # state_dict is tensors only — safe with weights_only=True below
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}, best val loss {best_val:.4f}")
                break
        if epoch % 30 == 0:
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load("export/transition_best.pt", weights_only=True))  # our own checkpoint, tensors only
    model.eval()

    # Bake de-normalization of the OUTPUT into the exported graph too, so the
    # browser only has to apply input normalization (meta.json), not invert
    # output normalization by hand — a wrapper module composes cleanly here
    # since y_mean/y_std are fixed constants at export time.
    class DenormalizedModel(nn.Module):
        def __init__(self, inner, y_mean, y_std):
            super().__init__()
            self.inner = inner
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))

        def forward(self, x):
            return self.inner(x) * self.y_std + self.y_mean

    export_ready = DenormalizedModel(model, y_mean, y_std)
    export_ready.eval()
    sample_input = torch.zeros(1, len(FEATURES))
    onnx_path, meta_path = export_model(
        export_ready, sample_input, "../public/host/models", "transition",
        FEATURES, x_mean.tolist(), x_std.tolist(),
    )
    print(f"Exported {onnx_path}, {meta_path}")
    print(f"Output order: {TARGETS}")


if __name__ == "__main__":
    main()
