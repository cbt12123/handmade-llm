"""A complete NumPy CNN: every derivative is explicit, without autograd."""
import argparse
import numpy as np
from sklearn.datasets import load_digits
from _common import OUT, split_indices, metrics, report, curves


def conv_forward(x, weight, bias):
    batch, _, height, width = x.shape
    channels, _, kh, kw = weight.shape
    out = np.empty((batch, channels, height-kh+1, width-kw+1))
    for row in range(out.shape[2]):
        for col in range(out.shape[3]):
            patch = x[:, :, row:row+kh, col:col+kw]
            out[:, :, row, col] = np.einsum("ncuv,fcuv->nf", patch, weight) + bias
    return out


def conv_backward(dout, x, weight):
    dx, dw = np.zeros_like(x), np.zeros_like(weight)
    kh, kw = weight.shape[2:]
    for row in range(dout.shape[2]):
        for col in range(dout.shape[3]):
            patch = x[:, :, row:row+kh, col:col+kw]
            signal = dout[:, :, row, col]
            dw += np.einsum("nf,ncuv->fcuv", signal, patch)
            dx[:, :, row:row+kh, col:col+kw] += np.einsum("nf,fcuv->ncuv", signal, weight)
    return dx, dw, dout.sum(axis=(0, 2, 3))


def pool_forward(x):
    n, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("This teaching implementation requires even spatial sizes.")
    windows = x.reshape(n, c, h//2, 2, w//2, 2).transpose(0, 1, 2, 4, 3, 5)
    windows = windows.reshape(n, c, h//2, w//2, 4)
    winners = windows.argmax(axis=-1)
    return windows.max(axis=-1), winners


def pool_backward(dout, winners):
    n, c, h, w = dout.shape
    windows = np.zeros((n, c, h, w, 4))
    np.put_along_axis(windows, winners[..., None], dout[..., None], axis=-1)
    return windows.reshape(n, c, h, w, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h*2, w*2)


def initialize(seed=42):
    rng = np.random.default_rng(seed)
    return {"kernel": rng.normal(0, np.sqrt(2/9), (8, 1, 3, 3)),
            "conv_bias": np.zeros(8),
            "linear": rng.normal(0, np.sqrt(2/72), (72, 10)),
            "linear_bias": np.zeros(10)}


def forward(x, p):
    z = conv_forward(x, p["kernel"], p["conv_bias"])
    pooled, winners = pool_forward(np.maximum(z, 0))
    flat = pooled.reshape(len(x), -1)
    logits = flat @ p["linear"] + p["linear_bias"]
    return logits, (x, z, pooled.shape, winners, flat)


def loss_and_gradient(x, y, p):
    logits, cache = forward(x, p)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=1, keepdims=True)
    loss = np.mean(np.log(exp.sum(axis=1)) - shifted[np.arange(len(y)), y])
    dlogits = prob.copy()
    dlogits[np.arange(len(y)), y] -= 1
    dlogits /= len(y)
    x, z, pool_shape, winners, flat = cache
    grads = {"linear": flat.T @ dlogits, "linear_bias": dlogits.sum(axis=0)}
    dpool = (dlogits @ p["linear"].T).reshape(pool_shape)
    dz = pool_backward(dpool, winners) * (z > 0)
    dx, grads["kernel"], grads["conv_bias"] = conv_backward(dz, x, p["kernel"])
    return float(loss), grads, dx


def check_derivatives():
    # Continuous random inputs avoid max-pool ties and ReLU kink points.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 1, 8, 8))
    y, p = np.array([2, 7]), initialize(7)
    _, grads, dx = loss_and_gradient(x, y, p)
    errors = {}
    for name, array, analytical in [(k, p[k], grads[k]) for k in p] + [("input", x, dx)]:
        maximum = 0.0
        for flat_index in rng.choice(array.size, min(12, array.size), replace=False):
            index = np.unravel_index(flat_index, array.shape)
            value = array[index]
            array[index] = value + 1e-5
            plus = loss_and_gradient(x, y, p)[0]
            array[index] = value - 1e-5
            minus = loss_and_gradient(x, y, p)[0]
            array[index] = value
            maximum = max(maximum, abs((plus-minus)/2e-5 - analytical[index]))
        errors[name] = maximum
    assert max(errors.values()) < 1e-6, errors
    # Independent implementation checks every derivative, including all pixels.
    import torch
    tx = torch.tensor(x, requires_grad=True)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}
    tz = torch.nn.functional.conv2d(tx, tp["kernel"], tp["conv_bias"])
    th = torch.nn.functional.max_pool2d(torch.relu(tz), 2).flatten(1)
    tl = th @ tp["linear"] + tp["linear_bias"]
    torch.nn.functional.cross_entropy(tl, torch.tensor(y, dtype=torch.long)).backward()
    reference = max([float(np.max(np.abs(tp[k].grad.numpy()-grads[k]))) for k in p]
                    + [float(np.max(np.abs(tx.grad.numpy()-dx)))])
    assert reference < 1e-10, reference
    return {"sampled_finite_difference_max_errors": errors,
            "all_derivatives_torch_reference_max_error": reference}


def evaluate(x, y, p):
    logits = forward(x, p)[0]
    shifted = logits-logits.max(axis=1, keepdims=True)
    loss = np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)), y])
    prediction = logits.argmax(axis=1)
    return float(loss), float(np.mean(prediction == y)), prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checks = check_derivatives()
    print(checks)
    if args.check_only:
        return
    data = load_digits()
    x, y = data.images[:, None].astype(np.float64)/16, data.target
    train, valid, test = split_indices(y)
    p, rng = initialize(), np.random.default_rng(42)
    history = {k: [] for k in ("train_loss", "valid_loss", "train_acc", "valid_acc")}
    best_loss, best, best_epoch = np.inf, None, None
    for epoch in range(1, 61):
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled), 64):
            batch = shuffled[start:start+64]
            _, grads, _ = loss_and_gradient(x[batch], y[batch], p)
            for name in p:
                p[name] -= 0.08 * grads[name]
        tr_loss, tr_acc, _ = evaluate(x[train], y[train], p)
        va_loss, va_acc, _ = evaluate(x[valid], y[valid], p)
        for key, value in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[key].append(value)
        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, epoch
            best = {k: v.copy() for k, v in p.items()}
        if epoch % 10 == 0:
            print(f"epoch={epoch}, train_loss={tr_loss:.4f}, valid_acc={va_acc:.4f}")
    np.savez(OUT/"11_numpy_best.npz", **best)
    with np.load(OUT/"11_numpy_best.npz") as saved:
        loaded = {k: saved[k].copy() for k in best}
    before = forward(x[test], best)[0]
    after = forward(x[test], loaded)[0]
    assert np.array_equal(before, after)
    _, _, prediction = evaluate(x[test], y[test], loaded)
    curves(history, "11_numpy_training.png")
    report("11_numpy_report.json", {"architecture": "1x8x8 -> conv8 valid3 -> ReLU -> pool2 -> flatten72 -> linear10",
           "gradient_checks": checks, "best_epoch": best_epoch, "history": history,
           "test": metrics(y[test], prediction), "reload_identical": True,
           "split_sizes": [len(train), len(valid), len(test)],
           "training": "NumPy only; SGD lr=0.08, batch=64, epochs=60; torch used only as derivative reference"})


if __name__ == "__main__":
    main()
