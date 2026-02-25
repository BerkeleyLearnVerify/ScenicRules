import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import choix
import numpy as np
from tqdm import tqdm

from reasonable_crowd.parse_trajectory import parse_trajectory
from rulebook_benchmark.process_trajectory import process_trajectory
from rulebook_benchmark.rulebook import Relation


def _parse_single_file(filename, trajectory_directory, network_U, network_S, step_size):
    traj_path = os.path.join(trajectory_directory, filename)
    if filename.startswith("U"):
        realization = parse_trajectory(traj_path, step_size=step_size)
        realization.network = network_U
    else:
        realization = parse_trajectory(traj_path, step_size=step_size)
        realization.network = network_S

    process_trajectory(realization)
    return filename, realization


def load_all_trajectories(
    trajectory_directory, network_U, network_S, step_size=100000, max_workers=8
):
    trajectory_files = [
        f for f in os.listdir(trajectory_directory) if f.endswith(".json")
    ]
    trajectories = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _parse_single_file,
                f,
                trajectory_directory,
                network_U,
                network_S,
                step_size,
            ): f
            for f in trajectory_files
        }

        with tqdm(total=len(futures), leave=False) as pbar:
            for future in as_completed(futures):
                filename, realization = future.result()
                trajectories.append((filename, realization))
                pbar.set_description(f"Parsing {filename}")
                pbar.update(1)

    return trajectories


def get_trajectories(output_directory, trajectory_directory, network_U, network_S):
    # check if trajectories.pkl exists
    if os.path.exists(os.path.join(output_directory, "trajectories.pkl")):
        print("Loading cached trajectories...")
        with open(os.path.join(output_directory, "trajectories.pkl"), "rb") as f:
            trajectories = pickle.load(f)

    else:
        print("Parsing trajectories from JSON files...")
        trajectories = load_all_trajectories(
            trajectory_directory,
            network_U,
            network_S,
            step_size=100000,
            max_workers=8,  # adjust depending on your CPU
        )
        os.makedirs(output_directory, exist_ok=True)
        with open(os.path.join(output_directory, "trajectories.pkl"), "wb") as f:
            pickle.dump(trajectories, f)

    return trajectories


def load_annotations(path_to_reasonable_crowd):
    annotation_file = os.path.join(
        path_to_reasonable_crowd, "annotations/annotations.json"
    )
    with open(annotation_file, "r") as f:
        data = json.load(f)
    f.close()
    return data


def build_evaluation_dataset(data):
    X, y, y_votes, y_agreement = [], [], [], []

    for scenario, annotations in data.items():
        # Collect all items and pairwise counts
        items = set()
        pair_counts = {}

        for pair, votes in annotations.items():
            t1, t2 = pair.split(" ;; ")
            items.update([t1, t2])
            pair_counts[(t1, t2)] = len(votes)

        items = sorted(items)
        id_to_idx = {item: i for i, item in enumerate(items)}
        n = len(items)
        comp_mat = np.zeros((n, n), dtype=int)

        for (t1, t2), count in pair_counts.items():
            i, j = id_to_idx[t1], id_to_idx[t2]
            comp_mat[i, j] = count

        scores = choix.ilsr_pairwise_dense(comp_mat + 1e-3, max_iter=1000, tol=1e-9)

        added = set()

        for (t1, t2), count12 in pair_counts.items():
            if (t1, t2) in added or (t2, t1) in added:
                continue
            if (t2, t1) in pair_counts:
                count21 = pair_counts[(t2, t1)]
                added.add((t1, t2))
                added.add((t2, t1))
            else:
                print(
                    "this should not happen: build_evaluation_dataset reverse pair missing"
                )
                assert False

            i, j = id_to_idx[t1], id_to_idx[t2]
            s1, s2 = scores[i], scores[j]

            if s1 > s2:
                human_pref = Relation.LARGER
            elif s1 < s2:
                human_pref = Relation.SMALLER
            else:
                print("this should not happen: build_evaluation_dataset equal scores")
                continue

            X.append((t1, t2))
            y.append(human_pref)
            y_votes.append((count12, count21))
            y_agreement.append(abs(count12 - count21) / (count12 + count21))

    return X, y, y_votes, y_agreement
