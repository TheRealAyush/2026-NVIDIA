import time
from typing import Tuple, List, Dict

import numpy as np

try:
    import cupy as cp
except Exception as e:
    raise RuntimeError(
        "CuPy is required for this script. Try: pip install cupy-cuda12x"
    ) from e


# -----------------------------
# LABS energy (CPU baseline)
# spins: 1D array of +/-1 of length N
# -----------------------------
def labs_energy_cpu(spins: np.ndarray) -> int:
    s = spins.astype(np.int32, copy=False)
    N = s.shape[0]
    E = 0
    for k in range(1, N):
        ck = int(np.dot(s[: N - k], s[k:]))
        E += ck * ck
    return int(E)


def labs_energy_cpu_batch(spins_batch: np.ndarray) -> np.ndarray:
    # spins_batch: shape (B, N)
    B, N = spins_batch.shape
    out = np.zeros((B,), dtype=np.int64)
    for b in range(B):
        out[b] = labs_energy_cpu(spins_batch[b])
    return out


# -----------------------------
# LABS energy (GPU batch)
# spins_batch: CuPy array shape (B, N), dtype int32, entries +/-1
# Computes energies for all B sequences in batch
# -----------------------------
def labs_energy_gpu_batch(spins_batch: "cp.ndarray") -> "cp.ndarray":
    # Ensure int32
    S = spins_batch.astype(cp.int32, copy=False)
    B, N = S.shape
    E = cp.zeros((B,), dtype=cp.int64)

    # For each k, compute ck for all batch members:
    # ck = sum_i S[:, i] * S[:, i+k]
    for k in range(1, N):
        ck = cp.sum(S[:, : N - k] * S[:, k:], axis=1, dtype=cp.int64)
        E += ck * ck
    return E


# -----------------------------
# Neighbor batch generator (single-bit flips)
# Given s (shape (N,)), produce B neighbors by flipping selected indices
# -----------------------------
def make_flip_neighbors(s: np.ndarray, flip_indices: np.ndarray) -> np.ndarray:
    # flip_indices: shape (B,), each entry in [0, N-1]
    B = flip_indices.shape[0]
    N = s.shape[0]
    neighbors = np.tile(s[None, :], (B, 1))
    neighbors[np.arange(B), flip_indices] *= -1
    return neighbors


# -----------------------------
# Benchmark helper
# -----------------------------
def time_fn(fn, *args, warmup: int = 1, iters: int = 5) -> float:
    # Warmup
    for _ in range(warmup):
        fn(*args)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def benchmark_one(N: int, B: int, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=(N,), replace=True).astype(np.int32)

    flip_idx = rng.integers(0, N, size=(B,), dtype=np.int32)
    neighbors_cpu = make_flip_neighbors(s, flip_idx).astype(np.int32)

    # CPU timing (batch loop)
    t_cpu = time_fn(labs_energy_cpu_batch, neighbors_cpu, warmup=1, iters=3)

    # GPU timing (single batch call)
    neighbors_gpu = cp.asarray(neighbors_cpu, dtype=cp.int32)

    # Make sure GPU is warmed up and synchronized
    _ = labs_energy_gpu_batch(neighbors_gpu)
    cp.cuda.runtime.deviceSynchronize()

    def gpu_call(x):
        y = labs_energy_gpu_batch(x)
        cp.cuda.runtime.deviceSynchronize()
        return y

    t_gpu = time_fn(gpu_call, neighbors_gpu, warmup=1, iters=5)

    # Speedup
    speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")

    return {
        "N": float(N),
        "B": float(B),
        "t_cpu_s": float(t_cpu),
        "t_gpu_s": float(t_gpu),
        "speedup": float(speedup),
    }


def main():
    # Choose sizes that will actually show GPU benefit
    Ns = [30, 50, 80, 100, 150]
    Bs = [256, 1024, 4096, 16384]

    print("GPU check:")
    print("  cupy devices:", cp.cuda.runtime.getDeviceCount())
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print("  device:", props.get("name", "unknown"))
    print()

    header = f"{'N':>5} {'B':>7} {'CPU (s)':>12} {'GPU (s)':>12} {'speedup':>10}"
    print(header)
    print("-" * len(header))

    results: List[Dict[str, float]] = []
    for N in Ns:
        for B in Bs:
            r = benchmark_one(N, B, seed=0)
            results.append(r)
            print(
                f"{int(r['N']):>5} {int(r['B']):>7} "
                f"{r['t_cpu_s']:>12.6f} {r['t_gpu_s']:>12.6f} {r['speedup']:>10.2f}"
            )

    # Optional: save a CSV for your report
    try:
        import csv

        out_path = "classical_gpu_bench.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["N", "B", "t_cpu_s", "t_gpu_s", "speedup"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print()
        print("Saved:", out_path)
    except Exception as e:
        print("Could not write CSV:", e)


if __name__ == "__main__":
    main()
