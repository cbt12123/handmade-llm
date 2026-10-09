"""Memory formulas and a small weight-only quantization experiment."""
import json
import numpy as np
from _common import model_path, save_report


def quantize(weight, per_row=False):
    maximum = np.max(np.abs(weight), axis=1, keepdims=True) if per_row else np.max(np.abs(weight))
    scale = np.where(maximum > 0, maximum/127, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(weight/scale), -127, 127).astype(np.int8)
    restored = quantized.astype(np.float32)*scale
    return quantized, scale, restored


def kv_bytes(layers, batch, sequence, kv_heads, head_dim, bytes_per_element):
    return 2*layers*batch*sequence*kv_heads*head_dim*bytes_per_element


def main():
    config = json.loads((model_path()/"config.json").read_text(encoding="utf-8"))
    head_dim = config["hidden_size"]//config["num_attention_heads"]
    budgets = []
    for length in [256, 1024, 4096, 8192]:
        for batch in [1,4]:
            size = kv_bytes(config["num_hidden_layers"],batch,length,config["num_key_value_heads"],head_dim,2)
            budgets.append({"sequence":length, "batch":batch, "fp16_kv_mib":size/2**20})
    rng = np.random.default_rng(42)
    weight = rng.normal(size=(8,16)).astype(np.float32)
    weight[0,0] = 40  # one outlier changes a global scale
    inputs = rng.normal(size=(5,16)).astype(np.float32)
    baseline = inputs @ weight.T
    results = {}
    for name, per_row in [("per_tensor",False),("per_output_row",True)]:
        q, scale, recovered = quantize(weight, per_row)
        output = inputs @ recovered.T
        assert q.dtype == np.int8 and np.isfinite(recovered).all()
        results[name] = {"weight_max_error":float(np.abs(weight-recovered).max()),
                         "output_rmse":float(np.sqrt(np.mean((baseline-output)**2))),
                         "float32_bytes":int(weight.nbytes), "int8_plus_scale_bytes":int(q.nbytes+scale.nbytes),
                         "scales":scale.reshape(-1).tolist()}
    zeros = quantize(np.zeros((2,3),np.float32), True)[2]
    assert np.array_equal(zeros, np.zeros_like(zeros))
    # Low-rank adapter algebra, not a claim of real pretrained-model fine-tuning.
    a, b = rng.normal(size=(16,2)), rng.normal(size=(2,8))
    scale = 4/2
    double_inputs, double_weight = inputs.astype(np.float64), weight.astype(np.float64)
    separate = double_inputs @ double_weight.T + scale*(double_inputs @ a) @ b
    merged = double_inputs @ (double_weight.T+scale*a@b)
    difference = float(np.abs(separate-merged).max())
    assert difference < 1e-10
    save_report("17_memory.json", {"kv_budget":budgets, "quantization":results,
            "lora_merge_max_error":difference,
            "scope":"Small matrix simulation only; no real model weights quantized or fine-tuned."})


if __name__ == "__main__":
    main()
