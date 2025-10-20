#!/usr/bin/env python3
"""
Example: Running benchmarks to compare Infinigram models.

Demonstrates how to use the evaluation framework to compare vanilla
Infinigram and RecursiveInfinigram on both in-distribution and OOD data.
"""

from infinigram.infinigram import Infinigram
from infinigram.recursive import (
    RecursiveInfinigram,
    CaseNormalizer,
    EditDistanceTransformer,
    SynonymTransformer
)
from infinigram.evaluation import (
    BenchmarkSuite,
    print_comparison_table,
    create_synthetic_corpus
)


def main():
    print("Infinigram Benchmark Suite")
    print("=" * 80)

    # Create a synthetic corpus
    print("\nCreating synthetic corpus...")
    corpus = create_synthetic_corpus(size=50000)
    print(f"Corpus size: {len(corpus)} bytes")

    # Create models
    print("\nInitializing models...")

    # Vanilla Infinigram
    vanilla = Infinigram(corpus)

    # RecursiveInfinigram with transformers
    recursive = RecursiveInfinigram(
        corpus,
        transformers=[
            CaseNormalizer(),
            EditDistanceTransformer(max_distance=1),
            SynonymTransformer(use_wordnet=True)
        ]
    )

    models = {
        "Vanilla Infinigram": vanilla,
        "Recursive Infinigram": recursive,
    }

    # Create benchmark suite
    print("\nCreating test datasets...")
    benchmark = BenchmarkSuite(corpus)

    # In-distribution test (easy)
    in_dist = benchmark.create_in_distribution_test(
        num_samples=200,
        context_length=20
    )

    # OOD test with case changes (medium)
    ood_case = benchmark.create_ood_test(
        transformations=['case'],
        num_samples=200,
        context_length=20
    )

    # OOD test with typos (hard)
    ood_typo = benchmark.create_ood_test(
        transformations=['typo'],
        num_samples=200,
        context_length=20
    )

    # OOD test with multiple transformations (very hard)
    ood_multi = benchmark.create_ood_test(
        transformations=['case', 'typo'],
        num_samples=200,
        context_length=20
    )

    test_datasets = {
        "In-Distribution": in_dist,
        "OOD: Case Changes": ood_case,
        "OOD: Typos": ood_typo,
        "OOD: Case + Typos": ood_multi,
    }

    # Run benchmarks
    print("\nRunning benchmarks...")
    print("(This may take a few minutes...)")

    results = benchmark.compare_models(
        models=models,
        test_datasets=test_datasets,
        top_k=10,
        verbose=True
    )

    # Print comparison table
    print_comparison_table(results)

    # Detailed analysis
    print("\n" + "=" * 80)
    print("Detailed Analysis")
    print("=" * 80)

    for dataset_name in test_datasets.keys():
        print(f"\n{dataset_name}:")

        for model_name in models.keys():
            metrics = results[model_name][dataset_name]
            print(f"\n  {model_name}:")
            print(f"    Accuracy:       {metrics.accuracy:.3f}")
            print(f"    Top-3 Accuracy: {metrics.top_k_accuracy.get(3, 0.0):.3f}")
            print(f"    Top-5 Accuracy: {metrics.top_k_accuracy.get(5, 0.0):.3f}")
            print(f"    Coverage:       {metrics.coverage:.3f}")
            print(f"    No Match Rate:  {metrics.no_match_rate:.3f}")
            print(f"    Perplexity:     {metrics.perplexity:.2f}")
            print(f"    Mean Prob:      {metrics.mean_probability:.4f}")
            print(f"    Mean Rank:      {metrics.mean_rank:.2f}")
            print(f"    Mean Time (ms): {metrics.mean_time_ms:.2f}")

    # Improvement analysis
    print("\n" + "=" * 80)
    print("Improvement Over Vanilla")
    print("=" * 80)

    for dataset_name in test_datasets.keys():
        vanilla_metrics = results["Vanilla Infinigram"][dataset_name]
        recursive_metrics = results["Recursive Infinigram"][dataset_name]

        acc_improvement = (recursive_metrics.accuracy - vanilla_metrics.accuracy) * 100
        cov_improvement = (recursive_metrics.coverage - vanilla_metrics.coverage) * 100

        print(f"\n{dataset_name}:")
        print(f"  Accuracy improvement:  {acc_improvement:+.1f}%")
        print(f"  Coverage improvement:  {cov_improvement:+.1f}%")

        if vanilla_metrics.perplexity < float('inf') and recursive_metrics.perplexity < float('inf'):
            ppl_improvement = (
                (vanilla_metrics.perplexity - recursive_metrics.perplexity) /
                vanilla_metrics.perplexity * 100
            )
            print(f"  Perplexity improvement: {ppl_improvement:+.1f}%")

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
