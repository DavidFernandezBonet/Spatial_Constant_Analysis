import networkx as nx
import matplotlib.pyplot as plt



### TODO: doesn't work for now
def draw_bfs(G, visited, queue, current, step):
    color_map = ['red' if node == current else
                 'blue' if node in visited else
                 'green' if node in queue else
                 'gray' for node in G]

    nx.draw(G, with_labels=True, node_color=color_map)
    plt.savefig(f"bfs_step_{step}.png")  # Save each step with a unique filename
    plt.close()



def bfs(G, start):
    visited = set()
    queue = [start]
    current = None
    states = []

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            queue.extend(set(G.neighbors(current)) - visited)

        # Save the state for animation
        states.append((set(visited), list(queue), current))

    return states


G = nx.grid_2d_graph(5, 5)

start_node = (0, 0)  # Starting node for BFS

bfs_states = bfs(G, start_node)

for step, state in enumerate(bfs_states):
    draw_bfs(G, *state, step)
