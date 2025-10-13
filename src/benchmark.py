import timeit

from typing import Any, List, Set, Union

import numpy as np

# Type hint for array or list of any type
ArrayLike = Union[List[Any], np.ndarray]

## --- FUNCTION DEFINITIONS ---


# 1. Original Function (Pure Python Loop, Slow)
def find_elements_original(data: ArrayLike, elements: ArrayLike) -> np.ndarray:
    """
    Slow Version: Pure Python iteration with O(N) lookup cost per item (if 'elements' is a list).
    Returns the index of the values from elements contained in array.
    """
    # Note: The 'in elements' check is the main bottleneck here.
    return np.array([idx for (idx, elem) in enumerate(data) if elem in elements], dtype=np.uint32)


# 2. Set Optimized Function (Python Loop, O(1) Lookup)
def find_elements_with_set(data: ArrayLike, elements: ArrayLike) -> np.ndarray:
    """
    Intermediate Version: Python iteration but with O(1) lookup using a Set.
    Improves lookup speed but still suffers from Python loop overhead.
    """
    # O(N) cost to create the set once, O(1) average cost for lookup.
    elements_set: Set[Any] = set(elements)

    return np.array([idx for (idx, elem) in enumerate(data) if elem in elements_set], dtype=np.uint32)


# 3. Optimized NumPy Function (Vectorized)
def find_elements_optimized(data: ArrayLike, elements: ArrayLike) -> np.ndarray:
    """
    Fastest Version (Consistent): Vectorized using np.isin() and np.where().
    Leverages optimized C-level code for maximum, consistent performance.
    """
    # Ensure data is a NumPy array
    data_array: np.ndarray = np.asarray(data)

    # 1. np.isin() creates a boolean mask
    mask: np.ndarray = np.isin(data_array, elements)

    # 2. np.where() efficiently extracts indices where the mask is True
    return np.where(mask)[0].astype(np.uint32)


# 4. Early Stop Function (Python Loop with Set and Break)
def find_elements_early_stop(data: ArrayLike, elements: ArrayLike) -> np.ndarray:
    """
    Context-Specific Version: Optimized to stop searching as soon as all 'elements'
    have been found (best for finding 'existence' at the start of the array).
    """
    # Copy and convert to a set for O(1) lookup and removal.
    lookups: Set[Any] = set(elements)
    data_indices: List[int] = []

    for idx, item in enumerate(data):
        if item in lookups:
            data_indices.append(idx)
            # Optimization: remove the item to speed up subsequent searches
            lookups.remove(item)

            # Optimization: break if no items are left to find
            if not lookups:
                break

    return np.array(data_indices, dtype=np.uint32)


## --- BENCHMARK SETUP ---


def run_benchmark():
    """Sets up the data and runs the performance comparison."""

    # PARAMETERS
    N_DATA = 1000000  # Total size of the array to search
    N_ELEMENTS = 1000  # Number of elements to find
    NUM_RUNS = 5  # Number of times to run the test for averaging
    np.random.seed(42)  # For reproducible results

    # DATA PREPARATION (Fixed targets)
    elements_to_find: np.ndarray = np.arange(0, N_ELEMENTS, 1)

    # SCENARIO 1: FAVORABLE (Targets found quickly at the beginning)
    data_favorable = np.arange(N_DATA)
    # Ensure the targets are among the first N_ELEMENTS items
    np.random.shuffle(data_favorable[:N_ELEMENTS])
    data_favorable_list = data_favorable.tolist()

    # SCENARIO 2: UNFAVORABLE (Targets found only at the end)
    data_unfavorable_list = data_favorable_list[::-1]

    elements_list = elements_to_find.tolist()

    print(f"--- Running Benchmark (N_DATA: {N_DATA}, N_ELEMENTS: {N_ELEMENTS}, Runs: {NUM_RUNS}) ---")

    # --- SCENARIO 1: FAVORABLE (Early Stop Wins) ---
    print("\n[SCENARIO 1: FAVORABLE (Targets at the start of the array)]")

    # 1. Original (List) - O(M*N) complexity
    time_original_fav = timeit.timeit(
        lambda: find_elements_original(data_favorable_list, elements_list),
        number=NUM_RUNS,
    )

    # 2. Set Optimized (Full Scan)
    time_set_fav = timeit.timeit(
        lambda: find_elements_with_set(data_favorable_list, elements_list),
        number=NUM_RUNS,
    )

    # 3. Early Stop (Stops quickly)
    time_early_stop_fav = timeit.timeit(
        lambda: find_elements_early_stop(data_favorable_list, elements_list),
        number=NUM_RUNS,
    )

    # 4. NumPy Optimized (Full Scan)
    time_optimized_fav = timeit.timeit(
        lambda: find_elements_optimized(data_favorable_list, elements_to_find),
        number=NUM_RUNS,
    )

    # Print Results for Scenario 1
    times_fav = {
        "1. Original (List)": time_original_fav / NUM_RUNS,
        "2. Set Optimized": time_set_fav / NUM_RUNS,
        "3. Early Stop": time_early_stop_fav / NUM_RUNS,
        "4. NumPy Optimized": time_optimized_fav / NUM_RUNS,
    }
    fastest_time_fav = min(times_fav.values())

    for name, t in times_fav.items():
        speed_up = fastest_time_fav / t
        print(f"  {name:<20}: {t:.6f}s (Factor: {1 / speed_up:.2f}x)")

    # --- SCENARIO 2: UNFAVORABLE (NumPy Wins) ---
    print("\n[SCENARIO 2: UNFAVORABLE (Targets at the end of the array)]")

    # 1. Original (List) - O(M*N) complexity
    time_original_unfav = timeit.timeit(
        lambda: find_elements_original(data_unfavorable_list, elements_list),
        number=NUM_RUNS,
    )

    # 2. Set Optimized (Full Scan)
    time_set_unfav = timeit.timeit(
        lambda: find_elements_with_set(data_unfavorable_list, elements_list),
        number=NUM_RUNS,
    )

    # 3. Early Stop (Forced full scan)
    time_early_stop_unfav = timeit.timeit(
        lambda: find_elements_early_stop(data_unfavorable_list, elements_list),
        number=NUM_RUNS,
    )

    # 4. NumPy Optimized (Full Scan)
    time_optimized_unfav = timeit.timeit(
        lambda: find_elements_optimized(data_unfavorable_list, elements_to_find),
        number=NUM_RUNS,
    )

    # Print Results for Scenario 2
    times_unfav = {
        "1. Original (List)": time_original_unfav / NUM_RUNS,
        "2. Set Optimized": time_set_unfav / NUM_RUNS,
        "3. Early Stop": time_early_stop_unfav / NUM_RUNS,
        "4. NumPy Optimized": time_optimized_unfav / NUM_RUNS,
    }
    fastest_time_unfav = min(times_unfav.values())

    for name, t in times_unfav.items():
        speed_up = fastest_time_unfav / t
        print(f"  {name:<20}: {t:.6f}s (Factor: {1 / speed_up:.2f}x)")


if __name__ == "__main__":
    run_benchmark()
