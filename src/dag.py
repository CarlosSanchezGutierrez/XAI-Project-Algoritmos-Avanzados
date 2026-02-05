import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make_random_dag(n_nodes: int, edge_prob: float, rng: np.random.Generator) -> nx.DiGraph:
    # Asegura aciclicidad: solo edges según un orden aleatorio, i -> j con i<j
    perm = rng.permutation(n_nodes)

    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                u = perm[i]
                v = perm[j]
                g.add_edge(u, v)

    mapping = {i: f"X{i}" for i in range(n_nodes)}
    g = nx.relabel_nodes(g, mapping)
    return g

def topo_order(g: nx.DiGraph):
    return list(nx.topological_sort(g))

def parents(g: nx.DiGraph, node: str):
    return list(g.predecessors(node))

def plot_dag(g: nx.DiGraph, title="Causal DAG"):
    plt.figure(figsize=(9, 6))

    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except:
        # fallback 100% python
        pos = nx.spring_layout(g, seed=42)

    nx.draw(
        g,
        pos,
        with_labels=True,
        node_size=1300,
        node_color="lightblue",
        edge_color="gray",
        arrows=True,
        font_size=10
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()