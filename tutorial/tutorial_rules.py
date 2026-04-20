from rulebook_benchmark.rulebook import Rule, Rulebook


def example_violation_function_1(handler, step):
    return 2


def example_violation_function_2(handler, step):
    return 100


def example_violation_function_3(handler, step):
    return step


example_rule_1 = Rule(example_violation_function_1, sum, "Example Rule 1", rule_id=1)
example_rule_2 = Rule(example_violation_function_2, max, "Example Rule 2", rule_id=2)
example_rule_3 = Rule(example_violation_function_3, max, "Example Rule 3", rule_id=3)


rule_id_to_rule = {
    example_rule_1.id: example_rule_1,
    example_rule_2.id: example_rule_2,
    example_rule_3.id: example_rule_3,
}

example_rulebook = Rulebook(rule_id_to_rule, rulebook_file="tutorial_rulebook.graph")
