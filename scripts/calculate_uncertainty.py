#!/usr/bin/env python3
"""Calculate spot- and sample-level clustering instability."""

from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import scanpy as sc


SAMPLES = ("151671", "151673")


def calculate_instability(partitions):
    """Exact instability calculation without an N x N co-association matrix."""

    R = len(partitions)
    if R < 2:
        raise ValueError("At least two clustering runs are required.")

    n = len(partitions[0])

    # Encode arbitrary cluster labels as compact integers.
    codes = []
    for labels in partitions:
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError("All partitions must contain the same spots.")
        codes.append(
            np.unique(labels, return_inverse=True)[1].astype(
                np.int32, copy=False
            )
        )

    if R * n >= np.iinfo(np.uint32).max:
        raise ValueError("n_runs * n_spots exceeds uint32 capacity.")

    row_sums = np.zeros(n, dtype=np.uint32)
    intra_sums = np.zeros((R, n), dtype=np.uint32)

    # Direct cluster-overlap calculation.
    for q, cq in enumerate(codes):
        sizes = np.bincount(cq).astype(np.uint32, copy=False)
        row_sums += sizes[cq]
        n_q = len(sizes)

        for r in range(q + 1):
            cr = codes[r]
            n_r = int(cr.max()) + 1

            if n_r * n_q <= np.iinfo(np.int32).max:
                key = cr * np.int32(n_q) + cq
            else:
                key = cr.astype(np.int64) * n_q + cq

            counts = np.bincount(key).astype(np.uint32, copy=False)
            contribution = counts[key] - np.uint32(1)

            intra_sums[r] += contribution
            if r != q:
                intra_sums[q] += contribution

    # LOO-normalized instability.
    row_sums = row_sums.astype(np.float32)
    spot_sum = np.zeros(n, dtype=np.float32)
    loo_factor = np.float32(R / (R - 1))

    for r, labels in enumerate(codes):
        raw_intra = intra_sums[r].astype(np.float32)

        intra_sum = raw_intra / R
        inter_sum = (row_sums - raw_intra - R) / R

        sizes = np.bincount(labels).astype(
            np.uint32, copy=False
        )[labels]

        intra_count = sizes - np.uint32(1)
        inter_count = np.uint32(n) - sizes

        # Singleton: mean_intra = 1.
        mean_intra = np.divide(
            intra_sum,
            intra_count,
            out=np.ones_like(intra_sum),
            where=intra_count != 0,
        )

        # Single global cluster: mean_inter = 0.
        mean_inter = np.divide(
            inter_sum,
            inter_count,
            out=np.zeros_like(inter_sum),
            where=inter_count != 0,
        )

        spot_sum += (
            1.0 - (mean_intra - mean_inter)
        ) * loo_factor

    spot_instability = spot_sum / np.float32(R)
    sample_instability = float(
        np.mean(spot_instability, dtype=np.float64)
    )

    return spot_instability, sample_instability


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--clustering-dir",
        type=Path,
        default=Path("results/leiden"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/uncertainty"),
    )
    parser.add_argument(
        "--run-counts",
        type=int,
        nargs="+",
        default=[20],
        help="Numbers of leading clustering runs to use.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        partition_path = (
            args.clustering_dir
            / f"{sample}_leiden_results_"
              f"{args.max_iterations}iterations.pkl"
        )

        partitions = joblib.load(partition_path)
        adata = sc.read_h5ad(
            args.data_dir / f"dlpfc_{sample}_filtered.h5ad"
        )

        for R in args.run_counts:
            if R < 2 or R > len(partitions):
                raise ValueError(
                    f"Requested {R} runs; "
                    f"{partition_path} contains {len(partitions)}."
                )

            spot, sample_score = calculate_instability(
                partitions[:R]
            )

            joblib.dump(
                {
                    "spot_uncertainty": spot,
                    "uncertainty": sample_score,
                    "iterations": R,
                },
                args.output_dir
                / f"dlpfc_{sample}_leiden_uncertainty_{R}runs.pkl",
            )

            key = f"leiden_uncertainty_{R}runs"
            adata.obs[key] = spot
            adata.uns[key] = sample_score

            print(
                f"{sample}, {R} runs: "
                f"mean instability={sample_score:.4f}"
            )

        adata.write_h5ad(
            args.output_dir
            / f"dlpfc_{sample}_leiden_uncertainty.h5ad"
        )


if __name__ == "__main__":
    main()
