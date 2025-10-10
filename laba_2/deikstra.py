import heapq
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

n = 10  # 10 cities
graph = defaultdict(list)

# Build line graph with distance 10 each
for i in range(9):
    w = 10
    graph[i].append((i + 1, w))
    graph[i + 1].append((i, w))

# Add shortcuts
graph[0].append((3, 25))
graph[3].append((0, 25))
graph[2].append((5, 20))
graph[5].append((2, 20))
graph[4].append((7, 30))
graph[7].append((4, 30))
graph[6].append((9, 25))
graph[9].append((6, 25))

mileage = 50
inf = float('inf')


# Modified Dijkstra function to track predecessors
def dijkstra(start):
    dist = [inf] * n
    dist[start] = 0
    pred = [None] * n  # Predecessor array
    pq = [(0, start)]  # (distance, city)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                pred[v] = u  # Update predecessor
                heapq.heappush(pq, (dist[v], v))
    return dist, pred


# Compute for each start
counts = {}
min_count = inf
min_cities = []
for s in range(n):
    dists, _ = dijkstra(s)
    count = sum(1 for d in dists if d <= mileage and d != inf)
    counts[s] = count
    if count < min_count:
        min_count = count
        min_cities = [s]
    elif count == min_count:
        min_cities.append(s)

# Output
print("Graph is a line 0-1-2-...-9 with each edge 10, plus shortcuts: 0-3:25, 2-5:20, 4-7:30, 6-9:25")
print(f"Mileage: {mileage}")
print("Reachable counts from each city:")
for s, c in counts.items():
    print(f"From {s}: {c}")
print(f"Minimal count: {min_count}, cities: {min_cities}")

# Visualization with shortest path from 0 to 9
G = nx.Graph()
for u in range(10):
    G.add_node(u)
for u in graph:
    for v, w in graph[u]:
        if u < v:  # avoid duplicates
            G.add_edge(u, v, weight=w)

# Get shortest path from node 0 to node 9
start_node = 0
end_node = 9
dists, pred = dijkstra(start_node)
# Reconstruct path from 0 to 9
path_nodes = []
path_edges = []
current = end_node
while current is not None and current != start_node:
    path_nodes.append(current)
    prev = pred[current]
    if prev is not None:
        # Add edge (min(prev, current), max(prev, current)) for undirected graph
        path_edges.append((min(prev, current), max(prev, current)))
    current = prev
if current == start_node:
    path_nodes.append(start_node)
path_nodes.reverse()  # Reverse to get path from 0 to 9

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original graph
pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax1)
ax1.set_title("Original Graph")

# Graph with highlighted shortest path from 0 to 9
nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax2)
# Highlight path nodes and edges
nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='lightgreen', node_size=500, ax=ax2)
nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='orange', node_size=700, ax=ax2)
nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color='purple', node_size=700, ax=ax2)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax2)
ax2.set_title(f"Shortest Path from Node {start_node} to Node {end_node} (Distance: {dists[end_node]})")

plt.tight_layout()
plt.show()