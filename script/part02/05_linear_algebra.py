"""Toy feature retrieval, transforms and SVD approximation."""
from _common import OUT, plt, np, write_json, transform_figure

names = ["记录A", "记录B", "记录C", "记录D"]
features = np.array([[1., 0., 0.], [.9, .1, 0.], [0., 1., 0.], [0., 0., 1.]])
query = np.array([1., .1, 0.])
norms = np.linalg.norm(features, axis=1)
assert np.all(norms > 0) and np.linalg.norm(query) > 0
similarities = features @ query / (norms * np.linalg.norm(query))
ranking = [{"name": names[i], "cosine": float(similarities[i])} for i in np.argsort(-similarities)]
assert ranking[0]["name"] == "记录B"
a = np.array([[1., 2., 3.], [2., 4.2, 6.], [3., 6., 9.3]])
u, s, vt = np.linalg.svd(a, full_matrices=False)
errors = {}
for rank in [1, 2]:
    approximate = (u[:, :rank] * s[:rank]) @ vt[:rank]
    errors[str(rank)] = float(np.linalg.norm(a - approximate))
assert errors["2"] <= errors["1"] + 1e-12
fig = transform_figure()
fig.savefig(OUT / "05-transforms.png", dpi=160)
plt.close(fig)
write_json("05-linear-algebra.json", {"query": query.tolist(), "features": features.tolist(), "ranking": ranking, "matrix_rank": int(np.linalg.matrix_rank(a)), "projection_rank": int(np.linalg.matrix_rank(np.diag([1., 0.]))), "approximation_errors": errors})
print("PASS: retrieval, rank and approximation checks")
