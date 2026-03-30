Evaluation against Human Preferences
=======================================
<!-- Outline:
- Brief intro to Reasonable crowd / Interface to Reasonable crowd
- EValuation process / options
- Try to describe in a generic way (i.e., not specific to reasonable crowd)
-->

This section outlines how to evaluate a rulebook's performance by comparing its automated judgments against human preferences, using the **Reasonable Crowd** dataset as a benchmark.

## Overview of Reasonable Crowd

**Reasonable Crowd** is a dataset designed to bridge the gap between formal safety rules and human intuition. It consists of numerous driving scenarios where human participants have provided votes on which of two trajectories is safer or more "reasonable". In the evaluation pipeline, the goal is to see if the rulebook ranks these scenarios in the same order as the majority of humans.

---

## The Evaluation Flow

The evaluation process follows a structured path from raw trajectory data to a final accuracy score:

1. Trajectories are loaded from `.json` files and mapped to their respective road networks (e.g., Town01, Town05).
    
2. Human votes and agreement levels are matched to the corresponding trajectory pairs.
    
3. A `Rulebook` is created using a `.graph` file (defining the hierarchy) and a mapping of IDs to `Rule` objects.
    
4. For each scenario pair (A,B), the rulebook evaluates both. If the rulebook finds more violations in A, it predicts B is safer.
    
5. The system calculates:
    
    - **Correct**: Rulebook and humans agree.
        
    - **Incomparable**: The rulebook cannot determine a winner (e.g., both have 0 violations).
        
    - **Accuracy**: The percentage of human-decided pairs correctly predicted by the rulebook.
        

---

## Caching and Optimization

Evaluating a complex rulebook across thousands of frames is computationally expensive. To solve this, the pipeline caches rule evaluations per trajectory-parameter pair:

- **Tuning Cache**: When a rule's parameters (like a distance threshold) are changed, the system checks a `tuning_cache.pkl`. If that specific rule with those specific parameters has been run on that trajectory before, it pulls the result instantly.
    
- **Optimization**: The optimization process aims to find the rulebook configuration that best matches human preferences (on the training split) by adjusting both rule parameters and rule priorities. The framework supports three primary optimization approaches:

Greedy Parameter Optimization: Sequentially adjusts parameters (like distance thresholds) for each rule to find a local optimum.

Greedy Group Optimization: Reorders entire categories of rules (e.g., swapping the priority of "Safety" vs. "Comfort" groups) to improve the overall match rate.

Brute Force Group Optimization: Exhaustively tests every possible permutation of rule groups to find the global optimum.
    
---

## How to Run the Script

To evaluate the rulebook against the dataset, ensure your directory structure matches the `path_to_reasonable_crowd` variable and run the evaluation script:

```
python get_all_results.py
```

## Extending to Other Datasets

While this pipeline is written for Reasonable Crowd, you can adapt it for any dataset by following these steps:

1. **Format Trajectories**: Ensure your data can be parsed into the `Realization` and `State` objects.
    
2. **Define Comparisons**: Create a mapping of "Scenario A vs Scenario B" with a ground truth label (1 if A is better, 0 if B is better).
    
3. **Implement `Rule` Functions**: Write functions that utilize the `VariableHandler` to extract the features relevant to your specific dataset.