"""Learning-rate comparison and finite-difference verification."""
from _common import OUT, plt, np, write_json

def loss(w):
    return (w - 3) ** 2

def derivative(w):
    return 2 * (w - 3)

records = {}
fig, ax = plt.subplots(figsize=(8, 4.5))
for lr in [.05, .4, 1.1]:
    w = 0.0
    parameters, losses = [w], [loss(w)]
    for _ in range(30):
        w -= lr * derivative(w)
        parameters.append(w)
        losses.append(loss(w))
    records[str(lr)] = {"parameters": parameters, "losses": losses}
    ax.semilogy(range(31), np.maximum(losses, 1e-15), label=f"lr={lr}")
assert records["0.4"]["losses"][-1] < records["0.05"]["losses"][-1]
assert records["1.1"]["losses"][-1] > records["1.1"]["losses"][0]
points = np.array([-2., 0., 1., 5.])
h = 1e-5
numerical = (loss(points+h) - loss(points-h)) / (2*h)
error = float(np.max(np.abs(numerical - derivative(points))))
assert error < 1e-6
ax.set(title="Gradient descent on (w - 3)^2", xlabel="Update step", ylabel="Loss (log scale; floor 1e-15)")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "04-learning-rates.png", dpi=160)
plt.close(fig)
write_json("04-calculus.json", {"target": 3, "derivative_max_error": error, "runs": records})
print("PASS: convergence, divergence and derivative checks")
