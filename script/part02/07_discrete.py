"""Compare minimum-edge BFS with weighted-DAG optimization."""
from collections import deque
from _common import OUT, plt, write_json, graph_figure

graph = {"A":[("B",2),("C",5)], "B":[("C",1),("D",4)], "C":[("D",1)], "D":[]}

def reconstruct(parent, end):
    if end not in parent:
        return None
    path = []
    while end is not None:
        path.append(end)
        end = parent[end]
    return path[::-1]

queue, parent = deque(["A"]), {"A":None}
while queue:
    node = queue.popleft()
    for neighbor, _ in graph[node]:
        if neighbor not in parent:
            parent[neighbor] = node
            queue.append(neighbor)
bfs_path = reconstruct(parent,"D")
assert len(bfs_path)-1 == 2

def optimal_path(edges):
    order = ["A","B","C","D"]
    positions = {node:i for i,node in enumerate(order)}
    assert all(positions[node] < positions[neighbor] for node in edges for neighbor,_ in edges[node])
    distance = {node:float("inf") for node in edges}
    distance["A"] = 0
    parent = {"A":None}
    for node in order:
        for neighbor,cost in edges[node]:
            candidate = distance[node]+cost
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                parent[neighbor] = node
    return reconstruct(parent,"D"), distance["D"]

best, cost = optimal_path(graph)
assert best == ["A","B","C","D"] and cost == 4
changed = {node:list(edges) for node,edges in graph.items()}
changed["B"] = [("C",1),("D",1)]
changed_path, changed_cost = optimal_path(changed)
assert changed_path == ["A","B","D"] and changed_cost == 3
dp = [1,1]
for step in range(2,7):
    dp.append(dp[-1]+dp[-2])
assert dp[0]==1 and dp[1]==1 and dp[6]==13
fig = graph_figure()
fig.savefig(OUT / "07-graph.png", dpi=160)
plt.close(fig)
write_json("07-discrete.json", {"bfs_path":bfs_path, "weighted_path":best, "weighted_cost":cost, "changed_path":changed_path, "changed_cost":changed_cost, "staircase_counts":dp})
print("PASS: BFS, weighted paths, changed-cost and boundary checks")
