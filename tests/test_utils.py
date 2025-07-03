"""Tests for polymer utility functions including SHI and nSHI calculations."""

import random

import networkx as nx
import pytest

from polymcsim.utils import calculate_nSHI, calculate_SHI

N = 1000
R = 0.8


@pytest.fixture
def simple_alternating_polymer_graph():
    """Create an alternating A-B-A-B polymer graph for testing."""
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, monomer_type="A" if i % 2 == 0 else "B")
        if i > 0:
            G.add_edges_from([(i - 1, i)])
    return G


@pytest.fixture
def simple_block_polymer_graph():
    """Create a block polymer graph (A block followed by B block)."""
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, monomer_type="A" if i < N / 2 else "B")
        if i > 0:
            G.add_edges_from([(i - 1, i)])
    return G


@pytest.fixture
def simple_random_polymer_graph():
    """Create a random 50:50 A:B polymer graph for testing."""
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, monomer_type=random.choice(["A", "B"]))
        if i > 0:
            G.add_edges_from([(i - 1, i)])
    return G


@pytest.fixture
def ratioed_block_polymer_graph():
    """Create a block polymer with custom ratio R."""
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, monomer_type="A" if i < N * R else "B")
        if i > 0:
            G.add_edges_from([(i - 1, i)])
    return G


@pytest.fixture
def ratioed_random_polymer_graph():
    """Create a random polymer with custom ratio R for A monomers."""
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, monomer_type="A" if random.random() < R else "B")
        if i > 0:
            G.add_edges_from([(i - 1, i)])
    return G


def test_calculate_SHI(
    simple_alternating_polymer_graph,
    simple_block_polymer_graph,
    simple_random_polymer_graph,
    ratioed_block_polymer_graph,
    ratioed_random_polymer_graph,
):
    """Test SHI calculation for a simple heterogeneous polymer."""
    # Create a simple polymer graph

    assert calculate_SHI(simple_alternating_polymer_graph) == pytest.approx(1, abs=1e-2)
    assert calculate_SHI(simple_block_polymer_graph) == pytest.approx(0, abs=1e-2)
    assert calculate_SHI(simple_random_polymer_graph) == pytest.approx(0.5, abs=1e-1)
    assert calculate_SHI(ratioed_block_polymer_graph) == pytest.approx(0, abs=1e-2)
    assert calculate_SHI(ratioed_random_polymer_graph) == pytest.approx(
        0.5 * R, abs=1e-1
    )


def test_calculate_nSHI(
    simple_alternating_polymer_graph,
    simple_block_polymer_graph,
    simple_random_polymer_graph,
    ratioed_block_polymer_graph,
    ratioed_random_polymer_graph,
):
    """Test SHI calculation for a simple heterogeneous polymer."""
    # Create a simple polymer graph

    assert calculate_nSHI(simple_alternating_polymer_graph) == pytest.approx(
        1, abs=1e-2
    )
    assert calculate_nSHI(simple_block_polymer_graph) == pytest.approx(0, abs=1e-2)
    assert calculate_nSHI(simple_random_polymer_graph) == pytest.approx(0.5, abs=1e-1)
    assert calculate_nSHI(ratioed_block_polymer_graph) == pytest.approx(0, abs=1e-2)
    assert calculate_nSHI(ratioed_random_polymer_graph) == pytest.approx(0.5, abs=1e-1)


def test_calculate_SHI_no_edges():
    """Test SHI calculation for a graph with no edges."""
    # Create a graph with no edges
    G = nx.Graph()
    G.add_node(0, monomer_type="A")
    G.add_node(1, monomer_type="B")

    # Expected SHI: 0.0
    assert calculate_SHI(G) == 0.0


def test_calculate_nSHI_less_than_2_nodes():
    """Test nSHI calculation for a graph with less than 2 nodes."""
    # Create a graph with one node
    G = nx.Graph()
    G.add_node(0, monomer_type="A")

    # Expected nSHI: 0.0
    assert calculate_nSHI(G) == 0.0


def test_calculate_nSHI_zero_expected_shi():
    """Test nSHI calculation when expected SHI is zero (all same monomer type)."""
    # Create a graph where all monomers are the same type
    G = nx.Graph()
    G.add_node(0, monomer_type="A")
    G.add_node(1, monomer_type="A")
    G.add_node(2, monomer_type="A")
    G.add_edges_from([(0, 1), (1, 2)])

    # Expected nSHI: 0.0
    assert calculate_nSHI(G) == 0.0
