"""Tests for polymer utility functions."""

import networkx as nx
import pytest

from polymcsim.utils import calculate_nSHI, calculate_SHI


def test_calculate_SHI():
    """Test SHI calculation for a simple heterogeneous polymer."""
    # Create a simple polymer graph
    G = nx.Graph()
    G.add_node(0, monomer_type="A")
    G.add_node(1, monomer_type="B")
    G.add_node(2, monomer_type="A")
    G.add_node(3, monomer_type="B")
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Expected SHI: 2/3
    expected_shi = 2 / 3
    assert calculate_SHI(G) == pytest.approx(expected_shi)


def test_calculate_nSHI():
    """Test nSHI calculation for a simple heterogeneous polymer."""
    # Create a simple polymer graph
    G = nx.Graph()
    G.add_node(0, monomer_type="A")
    G.add_node(1, monomer_type="B")
    G.add_node(2, monomer_type="A")
    G.add_node(3, monomer_type="B")
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Expected SHI: 2/3
    # Expected nSHI: (2/3) / (1 - (0.5^2 + 0.5^2)) = (2/3) / 0.5 = 4/3
    expected_nshi = 4 / 3
    assert calculate_nSHI(G) == pytest.approx(expected_nshi)


def test_calculate_SHI_no_edges():
    """Test SHI calculation for a graph with no edges."""
    # Create a graph with no edges
    G = nx.Graph()
    G.add_node(0, monomer_type="A")
    G.add_node(1, monomer_type="B")

    # Expected SHI: 0.0
    assert calculate_SHI(G) == 0.0


def test_calculate_nSHI_less_than_2_nodes():
    """Test nSHI calculation for a graph with fewer than 2 nodes."""
    # Create a graph with one node
    G = nx.Graph()
    G.add_node(0, monomer_type="A")

    # Expected nSHI: 0.0
    assert calculate_nSHI(G) == 0.0


def test_calculate_nSHI_zero_expected_shi():
    """Test nSHI calculation when expected SHI is zero (homopolymer)."""
    # Create a graph where all monomers are the same type
    G = nx.Graph()
    G.add_node(0, monomer_type="A")
    G.add_node(1, monomer_type="A")
    G.add_node(2, monomer_type="A")
    G.add_edges_from([(0, 1), (1, 2)])

    # Expected nSHI: 0.0
    assert calculate_nSHI(G) == 0.0
