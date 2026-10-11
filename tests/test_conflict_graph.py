"""Tests for conflict graph construction, graph properties, and drawing."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from exam_scheduler.conflict_graph import (
    build_conflict_graph,
    clique_lower_bound,
    conflict_edges,
    draw_conflict_graph,
)
from exam_scheduler.data import (
    generate_instance,
    ring_instance,
    sample_instance,
)


def test_sample_graph_structure():
    """Test conflict graph construction and isomorphism for sample instance."""
    inst = sample_instance()
    G = build_conflict_graph(inst)

    # Graph is isomorphic to 5-cycle graph
    assert nx.is_isomorphic(G, nx.cycle_graph(5))

    # Conflict edges match expected 5-cycle edges
    edges = conflict_edges(G)
    assert edges == [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]

    # Shared student edge attributes
    expected_shared = {(0, 1): 10, (0, 4): 9, (1, 2): 8, (2, 3): 12, (3, 4): 5}
    for u, v in edges:
        assert G.edges[u, v]["shared"] == expected_shared[(u, v)]


def test_clique_lower_bound():
    """Test clique lower bound calculation for 5-cycle and 3-ring (triangle)."""
    g_sample = build_conflict_graph(sample_instance())
    assert clique_lower_bound(g_sample) == 2

    g_ring3 = build_conflict_graph(ring_instance(3))
    assert clique_lower_bound(g_ring3) == 3


def test_generated_instance_graph_matches_shared_students():
    """Test generated instance graph edges match shared_students matrix."""
    inst = generate_instance(7, 50, seed=5)
    G = build_conflict_graph(inst)
    S = inst.shared_students()

    for i in range(inst.n_courses):
        for j in range(i + 1, inst.n_courses):
            if S[i, j] > 0:
                assert G.has_edge(i, j)
                assert G.edges[i, j]["shared"] == S[i, j]
            else:
                assert not G.has_edge(i, j)


def test_draw_conflict_graph_headless():
    """Test conflict graph rendering works headless without error."""
    G = build_conflict_graph(sample_instance())
    slot_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2}

    fig, ax = plt.subplots(figsize=(6, 6))
    res_ax = draw_conflict_graph(
        G, slot_of=slot_mapping, ax=ax, title="Test Headless Drawing"
    )
    assert res_ax is ax
    plt.close(fig)
