import os
from urllib.parse import quote, urlparse

import rdflib as rdf
from rdflib import XSD, Graph, Literal
from rdflib.graph import URIRef

from src.dglnn_local.RDFDataset import (AIFBDataset, AMDataset, BGSDataset,
                                        MUTAGDataset)

invalid_uri_chars = '<>" {}|\\^`'


def is_valid_uri(uri: str) -> bool:
    for c in invalid_uri_chars:
        if c in uri:
            return False
    return True


def check_data_dirs():
    data_paths = ["data", "data/KGs"]
    for path in data_paths:
        if not os.path.exists(path):
            os.makedirs(path)


def load_dataset(root="data/"):
    AIFBDataset(raw_dir=root)
    MUTAGDataset(raw_dir=root)
    BGSDataset(raw_dir=root)


def pre_process_mutag():
    """
    Processes the MUTAG dataset by filtering out certain predicates and blank nodes,
    and serializes the processed graph into a new file.

    The function reads the raw MUTAG dataset, filters out triples where the predicate
    is 'isMutagenic' and either the subject or object is a blank node. It also converts
    string literals to boolean True. The processed graph is then saved in a new file.
    """
    print("Preprocessing MUTAG")
    raw_path = "data/mutag-hetero_faec5b61/mutag_stripped.nt"
    processed_path = "data/KGs/mutag_stripped_processed.nt"

    # Check if the raw dataset file exists
    if os.path.isfile(raw_path):
        # Parse the raw graph
        g_mutag = Graph().parse(raw_path)

        # Initialize a new graph for the processed data
        g_mutag_new = Graph()
        is_mutagenic = rdf.term.URIRef(
            "http://dl-learner.org/carcinogenesis#isMutagenic"
        )
        BT = Literal(True, datatype=XSD.boolean)

        # Iterate over each triple in the graph
        for s, p, o in g_mutag:
            # Skip triples with 'isMutagenic' predicate
            if p == is_mutagenic:
                continue
            # Skip triples with blank nodes
            if isinstance(s, rdf.BNode) or isinstance(o, rdf.BNode):
                continue
            # Convert string literals to boolean True
            if (
                isinstance(o, rdf.Literal)
                and str(o.datatype) == "http://www.w3.org/2001/XMLSchema#string"
            ):
                g_mutag_new.add((s, p, BT))
                continue
            # Add the triple to the new graph
            g_mutag_new.add((s, p, o))

        # Serialize the new graph to a file
        g_mutag_new.serialize(destination=processed_path, encoding="utf-8", format="nt")
    else:
        print("Raw Dataset not Available")


def pre_process_aifb():
    """
    Processes the AIFB dataset by filtering out certain predicates and blank nodes,
    and serializes the processed graph into a new file.

    The function reads the raw AIFB dataset, filters out triples where the predicate
    is 'employs' or 'affiliation' and either the subject or object is a blank node.
    It also converts string literals to boolean True. The processed graph is then
    saved in a new file.
    """
    print("Preprocessing AIFB")
    raw_path = "data/aifb-hetero_82d021d8/aifbfixed_complete.n3"
    processed_path = "data/KGs/aifbfixed_complete_processed.n3"

    # Check if the raw dataset file exists
    if os.path.isfile(raw_path):
        # Parse the raw graph
        g_aifb = Graph().parse(raw_path)
        employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
        affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")
        BT = Literal(True, datatype=XSD.boolean)
        new_g_aifb = Graph()

        # Iterate over each triple in the graph
        for s, p, o in g_aifb:
            # Skip triples with 'employs' or 'affiliation' predicates
            if p == employs or p == affiliation:
                continue
            # Skip triples with blank nodes
            if isinstance(s, rdf.BNode) or isinstance(o, rdf.BNode):
                continue
            # Convert string literals to boolean True
            if (
                isinstance(o, rdf.Literal)
                and str(o.datatype) == "http://www.w3.org/2001/XMLSchema#string"
            ):
                new_g_aifb.add((s, p, BT))
                continue
            # Add the triple to the new graph
            new_g_aifb.add((s, p, o))

        # Serialize the new graph to a file
        new_g_aifb.serialize(destination=processed_path, encoding="utf-8", format="n3")
    else:
        print("Raw Dataset not Available")


def pre_process_bgs():
    print("Preprocessing BGS")
    g = Graph()
    g.parse("data/bgs-hetero_733c98ba/EarthMaterialClass_RockName.nt", format="nt")

    g2 = Graph()
    g2.parse("data/bgs-hetero_733c98ba/625KGeologyMap_Dyke.nt", format="nt")

    g3 = Graph()
    g3.parse("data/bgs-hetero_733c98ba/Lexicon_ShapeType.nt", format="nt")

    for s, p, o in g:
        if isinstance(o, URIRef):
            if not is_valid_uri(o):
                g.remove((s, p, o))

    for s, p, o in g2:
        if isinstance(o, URIRef):
            if not is_valid_uri(o):
                g2.remove((s, p, o))

    for s, p, o in g3:
        if isinstance(o, URIRef):
            if not is_valid_uri(o):
                g3.remove((s, p, o))

    g.serialize(
        "data/bgs-hetero_733c98ba/EarthMaterialClass_RockName.nt",
        format="nt",
        encoding="utf-8",
    )
    g2.serialize(
        "data/bgs-hetero_733c98ba/625KGeologyMap_Dyke.nt", format="nt", encoding="utf-8"
    )
    g3.serialize(
        "data/bgs-hetero_733c98ba/Lexicon_ShapeType.nt", format="nt", encoding="utf-8"
    )


load_dataset()
check_data_dirs()
pre_process_mutag()
pre_process_aifb()
pre_process_bgs()
#'Next Processing ---> Convert nt/n3 files to OWL KG Using ROBOT tool
# For AIFB remove the #Thing description from KG to make it compatible with EvoLearner as we get 'PSet Terminals have to have unique names
# As thing is already added from another instance, we can safely remove that manually to make it work
