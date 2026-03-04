"""
tests/test_graph_viz.py

Unit tests for galoop.graph_viz.
"""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from galoop.graph_viz import _element_color, build_page, generate_html


def _small_graph() -> nx.Graph:
    """Synthetic chem-env graph: Pt-Pt slab with an O-H adsorbate."""
    G = nx.Graph()
    # Slab nodes
    G.add_node("Pt:0[0,0,0]", index=0, ads=False, central_ads=False)
    G.add_node("Pt:1[0,0,0]", index=1, ads=False, central_ads=False)
    # Adsorbate nodes
    G.add_node("O:2[0,0,0]",  index=2, ads=True,  central_ads=True)
    G.add_node("H:3[0,0,0]",  index=3, ads=True,  central_ads=False)
    # Edges
    G.add_edge("Pt:0[0,0,0]", "Pt:1[0,0,0]", bond="PtPt", ads_only=2, dist=2)
    G.add_edge("Pt:0[0,0,0]", "O:2[0,0,0]",  bond="OPt",  ads_only=2, dist=1)
    G.add_edge("O:2[0,0,0]",  "H:3[0,0,0]",  bond="OH",   ads_only=0, dist=0)
    return G


def test_element_color_known():
    assert _element_color("O") == "#ff0d0d"
    assert _element_color("Pt") == "#d0d0e0"


def test_element_color_unknown():
    color = _element_color("Xx")
    assert color == "#aaaaaa"


def test_build_page_single_graph():
    G = _small_graph()
    page = build_page("Test page", [G])
    assert page["title"] == "Test page"
    assert isinstance(page["traces"], list)
    assert len(page["traces"]) > 0
    # Every trace must be a dict with 'type' key
    for trace in page["traces"]:
        assert isinstance(trace, dict)
        assert "type" in trace


def test_build_page_multiple_graphs():
    G = _small_graph()
    page = build_page("Multi", [G, G])
    assert len(page["traces"]) > 0


def test_generate_html_valid():
    G = _small_graph()
    pages = [
        build_page("Page 1", [G]),
        build_page("Page 2", [G]),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.html"
        generate_html(pages, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "Page 1" in content
        assert "Page 2" in content
        assert "ArrowLeft" in content    # keyboard navigation present
        assert "Plotly.react" in content


def test_generate_html_single_page():
    G = _small_graph()
    pages = [build_page("Only page", [G])]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "single.html"
        generate_html(pages, out)
        content = out.read_text(encoding="utf-8")
        assert "Only page" in content
        assert "1 / 1" in content
