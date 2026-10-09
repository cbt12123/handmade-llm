"""Coin simulation, Bayes, entropy and fixed-vocabulary smoothing."""
from collections import Counter
from _common import OUT, plt, np, write_json

rng = np.random.default_rng(42)
heads = rng.integers(0, 2, size=10000)
frequency = np.cumsum(heads) / np.arange(1, len(heads)+1)
assert abs(frequency[-1] - .5) < .03
posterior = (.9*.01) / (.9*.01 + .05*.99)
assert np.isclose(posterior, 90/585)

def entropy(p):
    p = np.asarray(p, dtype=float)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))

vocabulary = ["猫", "狗", "鸟", "鱼"]
counts = Counter(["猫", "狗", "猫", "鸟"])
test = ["猫", "鱼"]
unsmoothed = {word: counts[word]/4 for word in vocabulary}
smoothed = {word: (counts[word]+1)/(4+len(vocabulary)) for word in vocabulary}
assert np.isclose(sum(smoothed.values()), 1)
mean_loss = -np.mean([np.log(smoothed[word]) for word in test])
assert entropy([.5,.5]) > entropy([.9,.1])
fig, ax = plt.subplots(figsize=(8,4.5))
ax.plot(np.arange(1, len(heads)+1), frequency)
ax.axhline(.5, color="black", linestyle="--", label="P(heads)=0.5")
ax.set(title="Cumulative frequency", xlabel="Trials", ylabel="Fraction of heads", ylim=(0,1))
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "06-frequency.png", dpi=160)
plt.close(fig)
write_json("06-probability.json", {"seed":42, "trials":len(heads), "frequency":float(frequency[-1]), "posterior":posterior, "entropy_uniform":entropy([.5,.5]), "entropy_skewed":entropy([.9,.1]), "vocabulary":vocabulary, "test":test, "unsmoothed":unsmoothed, "unsmoothed_perplexity":"infinity: fish has probability zero", "smoothed":smoothed, "smoothed_perplexity":float(np.exp(mean_loss))})
print("PASS: probability normalization, posterior and entropy checks")
