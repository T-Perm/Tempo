import json

import numpy as np
import torch
import torch.nn as nn

from lib.onnx_export import export_model

FEATURES = ["candidateEnergy", "candidateBpm", "currentEnergy", "currentBpm", "energyTarget", "noveltyPenalty"]


class SelectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_dataset(path):
    with open(path) as f:
        rows = json.load(f)
    x = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float32)
    rank = np.array([r["rank"] for r in rows], dtype=np.float32)
    pick_id = np.array([r["pickId"] for r in rows], dtype=np.int64)
    return x, rank, pick_id


def build_groups(pick_id):
    """Returns a list of row-index arrays, one per distinct pickId, in
    first-occurrence order. Replaces fixed-GROUP_SIZE stride slicing now that
    group sizes vary (candidate pool shrinks within each simulated set)."""
    groups, order = {}, []
    for i, pid in enumerate(pick_id.tolist()):
        if pid not in groups:
            groups[pid] = []
            order.append(pid)
        groups[pid].append(i)
    return [np.array(groups[pid]) for pid in order]


def pairwise_margin_loss(scores, ranks, groups, n_pairs=4):
    """Within each group (rows sharing one pickId), sample n_pairs (better,
    worse) pairs by rank and penalize the model if it doesn't score the
    better one higher. `groups` here holds indices local to `scores`/`ranks`
    (already remapped for whichever subset — train or val — is passed in)."""
    loss = torch.tensor(0.0)
    count = 0
    for idx in groups:
        if len(idx) < 2:
            continue
        group_scores = scores[idx]
        group_ranks = ranks[idx]
        for _ in range(n_pairs):
            i, j = np.random.choice(len(idx), size=2, replace=False)
            if group_ranks[i] == group_ranks[j]:
                continue
            better, worse = (i, j) if group_ranks[i] < group_ranks[j] else (j, i)
            loss = loss + torch.clamp(1.0 - (group_scores[better] - group_scores[worse]), min=0)
            count += 1
    return loss / max(count, 1)


def split_train_val(groups, val_fraction=0.1, seed=0):
    """Splits by GROUP (pick), never by row — a group must never straddle
    train/val. Returns (train_row_idx, train_groups, val_row_idx, val_groups),
    where *_groups hold indices local to the *_row_idx-gathered subset."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_val = max(1, int(val_fraction * len(groups)))
    val_group_positions = set(order[:n_val].tolist())

    train_row_idx, val_row_idx = [], []
    train_groups, val_groups = [], []
    for gi, idx in enumerate(groups):
        if gi in val_group_positions:
            start = len(val_row_idx)
            val_row_idx.extend(idx.tolist())
            val_groups.append(np.arange(start, start + len(idx)))
        else:
            start = len(train_row_idx)
            train_row_idx.extend(idx.tolist())
            train_groups.append(np.arange(start, start + len(idx)))
    return train_row_idx, train_groups, val_row_idx, val_groups


def main():
    x, ranks, pick_id = load_dataset("data/selection_dataset.json")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x - mean) / std

    groups = build_groups(pick_id)
    train_row_idx, train_groups, val_row_idx, val_groups = split_train_val(groups)
    x_train, x_val = x_norm[train_row_idx], x_norm[val_row_idx]
    r_train, r_val = ranks[train_row_idx], ranks[val_row_idx]

    model = SelectionNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train_t = torch.tensor(x_train)
    r_train_t = torch.tensor(r_train)
    x_val_t = torch.tensor(x_val)
    r_val_t = torch.tensor(r_val)

    best_val = float("inf")
    patience, patience_left = 10, 10
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        scores = model(x_train_t)
        loss = pairwise_margin_loss(scores, r_train_t, train_groups)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = pairwise_margin_loss(model(x_val_t), r_val_t, val_groups).item()
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save(model.state_dict(), "export/selection_best.pt")  # state_dict is tensors only — safe with weights_only=True below
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}, best val loss {best_val:.4f}")
                break
        if epoch % 20 == 0:
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load("export/selection_best.pt", weights_only=True))  # our own checkpoint, tensors only
    model.eval()
    sample_input = torch.zeros(1, len(FEATURES))
    onnx_path, meta_path = export_model(
        model, sample_input, "../public/host/models", "selection",
        FEATURES, mean.tolist(), std.tolist(),
    )
    print(f"Exported {onnx_path}, {meta_path}")


if __name__ == "__main__":
    main()
