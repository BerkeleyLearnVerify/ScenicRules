import numpy as np

from rulebook_benchmark.realization import VariableHandler
from rulebook_benchmark.rulebook import Relation


def evaluate_rule_with_cache(
    rule, X, y, rule_parameter_result_dict, rule_id, trajectories_dict
):
    evaluations = {}

    # pbar = tqdm(total=len(trajectories_dict), desc="scoring trajectories",
    #           leave=False)
    for name, traj in trajectories_dict.items():
        evaluations[name] = rule.evaluate_with_cache(
            VariableHandler(traj), rule_parameter_result_dict, name, rule_id
        )

        # pbar.update(1)
    # pbar.close()

    correct = 0

    # pbar = tqdm(total=len(arr), desc="Evaluating rules", leave=False)
    for item in zip(X, y):
        (t1, t2), label = item
        r1 = evaluations[t1]
        r2 = evaluations[t2]
        if (
            (r1 < r2 and label == Relation.LARGER)
            or (r1 > r2 and label == Relation.SMALLER)
            or (r1 == r2 and label == Relation.EQUAL)
        ):
            correct += 1

    return correct


def evaluate_rulebook_with_cache(
    rulebook, X, y, y_votes, rule_parameter_result_dict, trajectories_dict
):
    evaluations = {}

    for name in trajectories_dict.keys():

        evaluations[name] = rulebook.evaluate_with_cache(
            rule_parameter_result_dict, name
        )

    correct = 0
    equal = 0
    incomparable = 0
    total = len(X)
    total_votes = 0
    correct_votes = 0
    reasons = []
    predictions = []

    for (t1, t2), label, (v1, v2) in zip(X, y, y_votes):
        r1 = evaluations[t1]
        r2 = evaluations[t2]

        model_pref, reason = rulebook.compare_results(r1, r2)
        if model_pref == label:
            correct += 1
            if model_pref == Relation.LARGER:
                correct_votes += v1
                total_votes += v1
            elif model_pref == Relation.SMALLER:
                correct_votes += v2
                total_votes += v2
        elif label == Relation.LARGER:
            total_votes += v1
        elif label == Relation.SMALLER:
            total_votes += v2

        if model_pref == Relation.NONCOMPARABLE:
            # print("Incomparable:", t1, t2, reason)
            incomparable += 1
        if model_pref == Relation.EQUAL:
            # print("Equal:", t1, t2, reason)
            equal += 1

        reasons.append(reason)
        predictions.append(model_pref)

    accuracy = correct / total
    weighted_accuracy = correct_votes / total_votes if total_votes > 0 else 0.0

    return (
        correct,
        equal,
        incomparable,
        total,
        accuracy,
        weighted_accuracy,
        reasons,
        predictions,
    )


def evaluate_rulebook(rulebook, X, y, y_votes, trajectories_dict):
    evaluations = {}

    # evaluate each trajectory without using any cache
    for name, realization in trajectories_dict.items():
        evaluations[name] = rulebook.evaluate(realization)

    correct = 0
    equal = 0
    incomparable = 0
    total = len(X)
    total_votes = 0
    correct_votes = 0
    reasons = []
    predictions = []

    for (t1, t2), label, (v1, v2) in zip(X, y, y_votes):
        r1 = evaluations[t1]
        r2 = evaluations[t2]

        model_pref, reason = rulebook.compare_trajectories(r1, r2)
        if model_pref == label:
            correct += 1
            if model_pref == Relation.LARGER:
                correct_votes += v1
                total_votes += v1
            elif model_pref == Relation.SMALLER:
                correct_votes += v2
                total_votes += v2
        elif label == Relation.LARGER:
            total_votes += v1
        elif label == Relation.SMALLER:
            total_votes += v2

        if model_pref == Relation.NONCOMPARABLE:
            incomparable += 1
        if model_pref == Relation.EQUAL:
            equal += 1

        reasons.append(reason)
        predictions.append(model_pref)

    accuracy = correct / total
    weighted_accuracy = correct_votes / total_votes if total_votes > 0 else 0.0

    return (
        correct,
        equal,
        incomparable,
        total,
        accuracy,
        weighted_accuracy,
        reasons,
        predictions,
    )
