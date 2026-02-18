def parse_log_file(log_path):
    results = {}
    unique_violations_global = set()

    current_data = {
        "errors": "",
        "rule_counts": "",
        "unique_violations": "",
        "failed": False,
    }
    failed_scenarios = []

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # Detect failure before results
        if "Exceeded maximum retries" in line:
            # Mark that the next scenario will be failed
            current_data["failed"] = True

        # Detect start of a scenario result block
        elif "Results for" in line and "with seed" in line:
            # Extract scenario name (e.g., common20_11)
            parts = line.split("Results for")[-1].split()
            scenario_name = parts[0].split(".scenic")[0]

        # Extract error line
        elif "Average error value" in line:
            parts = line.split("Average error value:")[-1].split(",")
            avg_error = parts[0].strip()
            avg_norm = parts[1].split(":")[-1].strip().split(",")[0]
            ce_ratio = line.split("Counterexample ratio:")[-1].split(",")[0].strip()
            max_error = line.split("Max error value:")[-1].split(",")[0].strip()
            max_norm = line.split("Max normalized error value:")[-1].strip()
            current_data["errors"] = (
                f"{avg_norm} {max_norm} {ce_ratio} {avg_error} {max_error}"
            )

        # Extract rule violation count
        elif "Rule violation count:" in line:
            dict_str = line.split("Rule violation count:")[-1].strip()
            dict_str = dict_str.strip("{}")
            pairs = dict_str.split(",")
            values = []
            for p in pairs:
                if ":" in p:
                    values.append(p.split(":")[-1].strip())
            current_data["rule_counts"] = " ".join(values)

        # Extract number of unique violations
        elif "Number of unique violations:" in line:
            num = line.split(":")[-1].strip()
            current_data["unique_violations"] = num

        # Collect all unique violations (for global count)
        elif "Unique violations:" in line:
            # Example: [['lane_centering', 'lane_keeping', 'reaching_goal']]
            uv_part = line.split("Unique violations:")[-1].strip()
            uv_part = uv_part.strip("[]")
            groups = uv_part.split("], [")
            for g in groups:
                g = (
                    g.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace('"', "")
                    .strip()
                )
                if g:
                    items = tuple(sorted(x.strip() for x in g.split(",") if x.strip()))
                    unique_violations_global.add(items)

        # Detect post-failure
        elif "❌ Scenario" in line and "exited with error code" in line:
            scenario_idx = line.split()[2].split(".scenic")[0].split("/")[-1]
            current_data["failed"] = True
            results[scenario_idx] = current_data
            current_data = {
                "errors": "",
                "rule_counts": "",
                "unique_violations": "",
                "failed": False,
            }

        elif "✅ Scenario" in line and "completed successfully" in line:
            scenario_idx = line.split()[2].split(".scenic")[0].split("/")[-1]
            results[scenario_idx] = current_data
            current_data = {
                "errors": "",
                "rule_counts": "",
                "unique_violations": "",
                "failed": False,
            }

    # Print results
    results = sorted(results.items())
    for scenario, data in results:
        if data["failed"]:
            print(f"{scenario}")
            failed_scenarios.append(scenario)
        else:
            print(
                f"{scenario} {data['errors']} {data['unique_violations']} {data['rule_counts']}"
            )
    print(
        f"\nTotal failed scenarios: {len(failed_scenarios)}, Failed indices: {failed_scenarios}"
    )

    # Global unique violation count
    print(
        f"\nTotal unique violations across all scenarios: {len(unique_violations_global)}"
    )


if __name__ == "__main__":
    log_file_path = "./outputs/collision_with_vru_ppo_eval.log"
    parse_log_file(log_file_path)
