import json
import os

import torch


def export_model(model, sample_input, out_dir, name, input_order, mean, std):
    """Exports a trained torch model to ONNX plus a sidecar .meta.json with
    the exact feature order and normalization stats the browser must apply
    before inference — the model's ONNX graph has no memory of feature names."""
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{name}.onnx")
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        input_names=["features"],
        output_names=["output"],
        dynamic_axes={"features": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    meta_path = os.path.join(out_dir, f"{name}-model.meta.json")
    with open(meta_path, "w") as f:
        json.dump({"inputOrder": input_order, "mean": mean, "std": std}, f, indent=2)
    return onnx_path, meta_path
