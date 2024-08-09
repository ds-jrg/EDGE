import os

from rdflib import Graph


def load_and_count_elements(file_path):
    """Loads an RDF graph from a file and counts unique elements manually."""
    graph = Graph()
    graph.parse(file_path)
    unique_elements = set()
    for s, p, o in graph:
        unique_elements.add(s)
        unique_elements.add(p)
        unique_elements.add(o)
    return len(unique_elements)


def assert_within_tolerance(actual, expected, tolerance_pct):
    """Asserts that 'actual' is within a percentage tolerance of 'expected'."""
    tolerance = expected * tolerance_pct / 100.0
    lower_bound = expected - tolerance
    upper_bound = expected + tolerance
    assert (
        lower_bound <= actual <= upper_bound
    ), f"Value {actual} not within {tolerance_pct}% of {expected}"


def test_aifb_graph_counts():
    """Tests if the AIFB graph count is within 0.1% tolerance of the expected count."""
    nt_count = load_and_count_elements("data/KGs/aifbfixed_complete_processed.n3")
    gt_count = load_and_count_elements("data/KGs/aifb.owl")

    assert_within_tolerance(nt_count, gt_count, 0.1)


def test_mutag_graph_counts():
    """Tests if the MUTAG graph count is within 0.1% tolerance of the expected count."""
    nt_count = load_and_count_elements("data/KGs/mutag_stripped_processed.nt")
    gt_count = load_and_count_elements("data/KGs/mutag.owl")

    assert_within_tolerance(nt_count, gt_count, 0.1)
