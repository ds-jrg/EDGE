import json
import os
import time

from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from ontolearn.refinement_operators import ModifiedCELOERefinement
from owlapy.model import IRI, OWLNamedIndividual
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calcuate_metrics(predictions_data):
    for _, examples in predictions_data.items():
        concept_individuals = set(examples["concept_individuals"])
        positive_examples_test = set(examples["positive_examples"])
        negative_examples_test = set(examples["negative_examples"])

        all_examples = positive_examples_test.union(negative_examples_test)

        true_labels = [
            1 if item in positive_examples_test else 0 for item in all_examples
        ]
        pred_labels = [1 if item in concept_individuals else 0 for item in all_examples]

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)

        # Calculate precision, recall, f1-score, and support for binary class
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="binary"
        )

    # Create a dictionary with metrics
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
    }

    return metrics_dict


def train_celoe(learning_problems, kg=None, use_heur=False):

    if learning_problems is None:
        print("No Learning problems provided. stopping training CELOE")
        return

    explanation_dict = {}
    kg_path = f"data/KGs/{kg}.owl"
    if not os.path.exists(kg_path):
        print(
            f"Dataset not found at: {kg_path}. Please  Provide dataset  {kg} at the designated location"
        )
        return None
    else:
        # Skip this dataset/model combination
        t0 = time.time()
        target_kb = KnowledgeBase(path=kg_path)
        for str_target_concept, examples in learning_problems.items():
            positive_examples = set(examples["positive_examples_train"])
            negative_examples = set(examples["negative_examples_train"])
            print("Target concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            if use_heur:
                qual = F1()
                heur = CELOEHeuristic(
                    expansionPenaltyFactor=0.05,
                    startNodeBonus=1.0,
                    nodeRefinementPenalty=0.01,
                )
                op = ModifiedCELOERefinement(
                    knowledge_base=target_kb,
                    use_negation=False,
                    use_all_constructor=False,
                )

                model = CELOE(
                    knowledge_base=target_kb,
                    max_runtime=600,
                    refinement_operator=op,
                    quality_func=qual,
                    heuristic_func=heur,
                    max_num_of_concepts_tested=10_000_000_000,
                    iter_bound=10_000_000_000,
                )
            else:
                model = CELOE(
                    knowledge_base=target_kb, quality_func=F1(), max_runtime=600
                )

            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            [print(_) for _ in hypotheses]
            positive_examples_test = set(examples["positive_examples_test"])
            negative_examples_test = set(examples["negative_examples_test"])
            best_concept = hypotheses[0].concept
            concept_ind = set(
                [
                    indv.get_iri().as_str()
                    for indv in target_kb.individuals_set(best_concept)
                ]
            )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            concept_inds = concept_ind.intersection(
                positive_examples_test | negative_examples_test
            )
            all_examples = positive_examples_test.union(negative_examples_test)
            predicitons_dict = {
                item: 1 if item in concept_inds else 0 for item in all_examples
            }
            explanation_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
            }

        t1 = time.time()
        duration = t1 - t0
        return predicitons_dict, duration, explanation_dict
