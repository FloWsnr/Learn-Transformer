"""
Asyncio Practice Task: Async Model Evaluation Pipeline
=======================================================

Scenario: You're building a pipeline that evaluates multiple ML models
against multiple datasets concurrently. Each evaluation involves:
1. Fetching the model (slow I/O - simulated)
2. Fetching the dataset (slow I/O - simulated)
3. Running evaluation (CPU-bound-ish, simulated)
4. Collecting all results

Your job: implement the functions marked with TODO.
The goal is to run things concurrently where possible.

Run:  python asyncio_practice.py
Test: python asyncio_practice.py --solution
"""

import asyncio
import time
import random
import argparse

# ── Simulated async I/O (don't modify) ──────────────────────────────


async def _fetch_model(model_name: str) -> dict:
    """Simulates downloading a model. Takes 0.5-1.5s."""
    delay = random.uniform(0.5, 1.5)
    await asyncio.sleep(delay)
    return {"name": model_name, "params": random.randint(1, 100) * 1_000_000}


async def _fetch_dataset(dataset_name: str) -> dict:
    """Simulates loading a dataset. Takes 0.3-1.0s."""
    delay = random.uniform(0.3, 1.0)
    await asyncio.sleep(delay)
    return {"name": dataset_name, "size": random.randint(1000, 50000)}


async def _run_eval(model: dict, dataset: dict) -> dict:
    """Simulates running evaluation. Takes 0.2-0.5s."""
    delay = random.uniform(0.2, 0.5)
    await asyncio.sleep(delay)
    score = random.uniform(0.5, 0.99)
    return {
        "model": model["name"],
        "dataset": dataset["name"],
        "score": round(score, 4),
    }


# ── Task 1: Basic concurrency ───────────────────────────────────────


async def fetch_all_models(model_names: list[str]) -> list[dict]:
    """
    TODO: Fetch all models concurrently (not sequentially!).
    Return a list of model dicts in the same order as model_names.

    Hint: asyncio.gather()
    """
    results = await asyncio.gather(*[_fetch_model(m) for m in model_names])
    return results


async def fetch_all_datasets(dataset_names: list[str]) -> list[dict]:
    """
    TODO: Fetch all datasets concurrently.
    Return a list of dataset dicts in the same order as dataset_names.
    """
    results = await asyncio.gather(*[_fetch_dataset(dn) for dn in dataset_names])
    return results


# ── Task 2: Evaluate one model against one dataset ──────────────────


async def evaluate_single(model_name: str, dataset_name: str) -> dict:
    """
    TODO: Fetch the model and dataset concurrently,
    then run evaluation on them. Return the eval result.

    Steps:
    1. Fetch model and dataset at the same time (not one after another)
    2. Once both are ready, run _run_eval
    3. Return the result
    """
    coros = [_fetch_model(model_name), _fetch_dataset(dataset_name)]
    results = await asyncio.gather(*coros)
    result = _run_eval(results[0], results[1])
    return await result


# ── Task 3: Full pipeline with concurrency limit ────────────────────


async def evaluate_all(
    model_names: list[str],
    dataset_names: list[str],
    max_concurrent: int = 3,
) -> list[dict]:
    """
    TODO: Evaluate every (model, dataset) pair concurrently,
    but limit to max_concurrent evaluations running at once.

    Hint: asyncio.Semaphore

    Return a list of all result dicts.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def sem_eval_single(m_name, ds_name):
        async with sem:
            return await evaluate_single(m_name, ds_name)

    tasks = [sem_eval_single(m, ds) for m in model_names for ds in dataset_names]
    results = await asyncio.gather(*tasks)

    return results


# ── Task 4: Timeout handling ─────────────────────────────────────────


async def fetch_with_timeout(
    model_name: str, timeout_seconds: float = 1.0
) -> dict | None:
    """
    TODO: Fetch a model, but return None if it takes longer than
    timeout_seconds. Don't let it raise an exception.

    Hint: asyncio.wait_for() + try/except asyncio.TimeoutError
    """
    raise NotImplementedError


# ── Task 5: Producer-consumer with asyncio.Queue ─────────────────────


async def producer(
    queue: asyncio.Queue,
    model_names: list[str],
    dataset_names: list[str],
):
    """
    TODO: Put all (model_name, dataset_name) pairs into the queue.
    After all pairs are added, put a None sentinel for each consumer
    to signal they should stop.

    For this task, assume 2 consumers.
    """
    raise NotImplementedError


async def consumer(queue: asyncio.Queue, results: list[dict], consumer_id: int):
    """
    TODO: Continuously get items from the queue, run evaluate_single,
    and append results. Stop when you receive None.
    Print a message like "Consumer {consumer_id} processing {model} x {dataset}"
    """
    raise NotImplementedError


async def run_pipeline_with_queue(
    model_names: list[str], dataset_names: list[str]
) -> list[dict]:
    """
    TODO: Set up a queue, 1 producer, and 2 consumers.
    Run them all concurrently and return collected results.

    Hint: asyncio.gather() the producer and consumers together.
    """
    raise NotImplementedError


# =====================================================================
# SOLUTIONS (don't peek until you've tried!)
# =====================================================================


async def _sol_fetch_all_models(model_names):
    return await asyncio.gather(*[_fetch_model(name) for name in model_names])


async def _sol_fetch_all_datasets(dataset_names):
    return await asyncio.gather(*[_fetch_dataset(name) for name in dataset_names])


async def _sol_evaluate_single(model_name, dataset_name):
    model, dataset = await asyncio.gather(
        _fetch_model(model_name),
        _fetch_dataset(dataset_name),
    )
    return await _run_eval(model, dataset)


async def _sol_evaluate_all(model_names, dataset_names, max_concurrent=3):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_eval(m, d):
        async with semaphore:
            return await _sol_evaluate_single(m, d)

    tasks = [limited_eval(m, d) for m in model_names for d in dataset_names]
    return await asyncio.gather(*tasks)


async def _sol_fetch_with_timeout(model_name, timeout_seconds=1.0):
    try:
        return await asyncio.wait_for(_fetch_model(model_name), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"  Timeout fetching {model_name}")
        return None


async def _sol_producer(queue, model_names, dataset_names):
    for m in model_names:
        for d in dataset_names:
            await queue.put((m, d))
    # Sentinel for each consumer
    for _ in range(2):
        await queue.put(None)


async def _sol_consumer(queue, results, consumer_id):
    while True:
        item = await queue.get()
        if item is None:
            break
        model_name, dataset_name = item
        print(f"  Consumer {consumer_id} processing {model_name} x {dataset_name}")
        result = await _sol_evaluate_single(model_name, dataset_name)
        results.append(result)
        queue.task_done()


async def _sol_run_pipeline_with_queue(model_names, dataset_names):
    queue = asyncio.Queue()
    results = []
    await asyncio.gather(
        _sol_producer(queue, model_names, dataset_names),
        _sol_consumer(queue, results, 1),
        _sol_consumer(queue, results, 2),
    )
    return results


# =====================================================================
# TEST RUNNER
# =====================================================================


async def run_tests(use_solutions: bool = False):
    models = ["mistral-7b", "mistral-8x7b", "pixtral-12b"]
    datasets = ["mmlu", "hellaswag", "arc"]

    if use_solutions:
        _fetch_all_models = _sol_fetch_all_models
        _fetch_all_datasets = _sol_fetch_all_datasets
        _evaluate_single = _sol_evaluate_single
        _evaluate_all = _sol_evaluate_all
        _fetch_with_timeout = _sol_fetch_with_timeout
        _run_pipeline = _sol_run_pipeline_with_queue
    else:
        _fetch_all_models = fetch_all_models
        _fetch_all_datasets = fetch_all_datasets
        _evaluate_single = evaluate_single
        _evaluate_all = evaluate_all
        _fetch_with_timeout = fetch_with_timeout
        _run_pipeline = run_pipeline_with_queue

    # Test 1
    print("=" * 60)
    print("Task 1: Fetch all models & datasets concurrently")
    print("=" * 60)
    t0 = time.perf_counter()
    fetched_models = await _fetch_all_models(models)
    elapsed = time.perf_counter() - t0
    print(f"  Fetched {len(fetched_models)} models in {elapsed:.2f}s")
    print(f"  (Sequential would take ~3s, concurrent should be ~1.5s)")
    assert len(fetched_models) == 3
    assert all("name" in m for m in fetched_models)
    print("  ✓ Passed\n")

    # Test 2
    print("=" * 60)
    print("Task 2: Evaluate single (model + dataset fetched concurrently)")
    print("=" * 60)
    t0 = time.perf_counter()
    result = await _evaluate_single("mistral-7b", "mmlu")
    elapsed = time.perf_counter() - t0
    print(f"  Result: {result}")
    print(f"  Took {elapsed:.2f}s (should be ~1.5-2s, not ~2.5-3s)")
    assert "score" in result
    assert result["model"] == "mistral-7b"
    print("  ✓ Passed\n")

    # Test 3
    print("=" * 60)
    print("Task 3: Evaluate all pairs with concurrency limit")
    print("=" * 60)
    t0 = time.perf_counter()
    results = await _evaluate_all(models, datasets, max_concurrent=3)
    elapsed = time.perf_counter() - t0
    print(f"  Got {len(results)} results in {elapsed:.2f}s")
    print(f"  (9 pairs, 3 concurrent → should be ~3-5s, not ~15s)")
    assert len(results) == 9
    print("  ✓ Passed\n")

    # Test 4
    print("=" * 60)
    print("Task 4: Fetch with timeout")
    print("=" * 60)
    successes, timeouts = 0, 0
    for _ in range(5):
        r = await _fetch_with_timeout("test-model", timeout_seconds=0.8)
        if r is None:
            timeouts += 1
        else:
            successes += 1
    print(
        f"  5 attempts with 0.8s timeout: {successes} succeeded, {timeouts} timed out"
    )
    print("  ✓ Passed (both outcomes are valid due to random delays)\n")

    # Test 5
    print("=" * 60)
    print("Task 5: Producer-consumer pipeline")
    print("=" * 60)
    t0 = time.perf_counter()
    results = await _run_pipeline(models[:2], datasets[:2])
    elapsed = time.perf_counter() - t0
    print(f"  Got {len(results)} results in {elapsed:.2f}s")
    assert len(results) == 4  # 2 models x 2 datasets
    for r in results:
        print(f"    {r['model']} x {r['dataset']}: {r['score']}")
    print("  ✓ Passed\n")

    print("=" * 60)
    print("ALL TASKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solution", action="store_true", help="Run with reference solutions"
    )
    args = parser.parse_args()
    asyncio.run(run_tests(use_solutions=args.solution))
