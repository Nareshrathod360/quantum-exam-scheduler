"""Conflict graph construction and visualization module."""

from typing import Dict, List, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from exam_scheduler.data import ExamInstance


def build_conflict_graph(instance: ExamInstance) -> nx.Graph:
    """Build a NetworkX graph representing course conflict topology."""
    G = nx.Graph()
    counts = instance.enrollment_counts()
    for i in range(instance.n_courses):
        G.add_node(
            i,
            code=instance.course_codes[i],
            title=instance.course_titles[i],
            students=int(counts[i]),
        )

    S = instance.shared_students()
    n = instance.n_courses
    for i in range(n):
        for j in range(i + 1, n):
            shared_count = int(S[i, j])
            if shared_count > 0:
                G.add_edge(i, j, shared=shared_count)

    return G


def conflict_edges(graph: nx.Graph) -> List[Tuple[int, int]]:
    """Return sorted list of (i, j) edges with i < j."""
    edges = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    return sorted(edges)


def clique_lower_bound(graph: nx.Graph) -> int:
    """Return the size of the largest clique in the graph."""
    if graph.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.find_cliques(graph)), default=0)


def draw_conflict_graph(
    graph: nx.Graph,
    slot_of: Optional[Dict[int, int]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Draw conflict graph with circular layout and optional slot coloring."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    pos = nx.circular_layout(graph)
    node_labels = {n: graph.nodes[n].get("code", str(n)) for n in graph.nodes()}
    edge_labels = {
        (u, v): graph.edges[u, v]["shared"]
        for u, v in graph.edges()
        if "shared" in graph.edges[u, v]
    }

    if slot_of is not None:
        cmap = matplotlib.colormaps["tab10"]
        node_colors = [cmap(slot_of.get(n, 0) % 10) for n in graph.nodes()]
    else:
        node_colors = "skyblue"

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(
        graph, pos, labels=node_labels, ax=ax, font_size=10, font_weight="bold"
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="gray", width=1.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=9)

    if title:
        ax.set_title(title)

    ax.axis("off")
    return ax
