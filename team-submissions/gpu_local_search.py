import time
import numpy as np
import cupy as cp

def labs_energy_gpu_batch(spins_batch: "cp.ndarray") -> "cp.ndarray":
    S = spins_batch.astype(cp.int32, copy=False)
    B, N = S.shape
    E = cp.zeros((B,), dtype=cp.int64)
    for k in range(1, N):
        ck = cp.sum(S[:, : N - k] * S[:, k:], axis=1, dtype=cp.int64)
        E += ck * ck
    return E

def labs_energy_cpu(spins: np.ndarray) -> int:
    s = spins.astype(np.int32, copy=False)
    N = s.shape[0]
    E = 0
    for k in range(1, N):
        ck = int(np.dot(s[: N - k], s[k:]))
        E += ck * ck
    return int(E)

def make_flip_neighbors(s: np.ndarray, flip_indices: np.ndarray) -> np.ndarray:
    B = flip_indices.shape[0]
    neighbors = np.tile(s[None, :], (B, 1))
    neighbors[np.arange(B), flip_indices] *= -1
    return neighbors

def gpu_best_neighbor_step(s: np.ndarray, B: int, rng: np.random.Generator):
    N = s.shape[0]
    flips = rng.integers(0, N, size=(B,), dtype=np.int32)
    neigh = make_flip_neighbors(s, flips).astype(np.int32)
    neigh_gpu = cp.asarray(neigh, dtype=cp.int32)

    E = labs_energy_gpu_batch(neigh_gpu)
    cp.cuda.runtime.deviceSynchronize()

    best_i = int(cp.argmin(E).get())
    best_s = neigh[best_i]
    best_e = int(E[best_i].get())
    return best_s, best_e

def cpu_best_neighbor_step(s: np.ndarray, B: int, rng: np.random.Generator):
    N = s.shape[0]
    flips = rng.integers(0, N, size=(B,), dtype=np.int32)
    neigh = make_flip_neighbors(s, flips).astype(np.int32)

    best_e = None
    best_s = None
    for i in range(B):
        e = labs_energy_cpu(neigh[i])
        if best_e is None or e < best_e:
            best_e = e
            best_s = neigh[i]
    return best_s, int(best_e)

def run_search(N=100, B=4096, steps=50, seed=0):
    rng = np.random.default_rng(seed)
    s0 = rng.choice([-1, 1], size=(N,), replace=True).astype(np.int32)

    # GPU run
    s = s0.copy()
    best_e_gpu = labs_energy_cpu(s)
    t0 = time.perf_counter()
    for _ in range(steps):
        cand_s, cand_e = gpu_best_neighbor_step(s, B, rng)
        if cand_e <= best_e_gpu:
            s = cand_s
            best_e_gpu = cand_e
    t1 = time.perf_counter()

    # CPU run
    s = s0.copy()
    best_e_cpu = labs_energy_cpu(s)
    t2 = time.perf_counter()
    for _ in range(steps):
        cand_s, cand_e = cpu_best_neighbor_step(s, B, rng)
        if cand_e <= best_e_cpu:
            s = cand_s
            best_e_cpu = cand_e
    t3 = time.perf_counter()

    print("Config:", f"N={N} B={B} steps={steps}")
    print("GPU search:", f"time={t1-t0:.3f}s", f"bestE={best_e_gpu}")
    print("CPU search:", f"time={t3-t2:.3f}s", f"bestE={best_e_cpu}")
    if (t1 - t0) > 0:
        print("Speedup:", f"{(t3-t2)/(t1-t0):.2f}x")

def main():
    # Pick settings that show huge benefit
    run_search(N=80,  B=4096,  steps=30, seed=0)
    run_search(N=100, B=4096,  steps=30, seed=0)
    run_search(N=150, B=16384, steps=20, seed=0)

if __name__ == "__main__":
    main()
