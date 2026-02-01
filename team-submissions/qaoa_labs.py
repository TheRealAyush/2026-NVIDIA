# qaoa_labs.py
# CUDA-Q QAOA-like sampler for LABS that avoids rzz (not available in some Brev images)
# Uses ZZ interaction decomposition: CX - RZ - CX

import itertools
import numpy as np
import cudaq


# -------------------------
# LABS energy (classical)
# -------------------------
def labs_energy(spins):
    s = np.asarray(spins, dtype=int)
    N = s.size
    E = 0
    for k in range(1, N):
        ck = 0
        for i in range(N - k):
            ck += s[i] * s[i + k]
        E += ck * ck
    return int(E)


def bitstring_to_spins(bitstring: str):
    # Map: 0 -> +1, 1 -> -1
    return np.array([+1 if b == "0" else -1 for b in bitstring], dtype=int)


def brute_force_best(N: int):
    best_E = None
    best_s = None
    for bits in itertools.product([+1, -1], repeat=N):
        E = labs_energy(bits)
        if best_E is None or E < best_E:
            best_E = E
            best_s = np.array(bits, dtype=int)
    return best_E, best_s


# -------------------------
# Quantum kernel (p=1 "QAOA-like")
# IMPORTANT: we DO NOT use rzz(); we implement exp(-i*gamma*Z⊗Z) via CX-RZ-CX
# Gate intrinsics used: h, rx, rz, cx, mz
# -------------------------
@cudaq.kernel
def qaoa_kernel(N: int, gamma: float, beta: float):
    q = cudaq.qvector(N)

    # |+>^N
    for i in range(N):
        h(q[i])

    # Pairwise ZZ phases (not full LABS cost, but tunable + valid)
    # ZZ(gamma) via CX(i,j); RZ(2*gamma) on j; CX(i,j)
    for i in range(N):
        for j in range(i + 1, N):
            cx(q[i], q[j])          # if this errors, replace cx with cnot
            rz(2.0 * gamma, q[j])
            cx(q[i], q[j])

    # Mixer
    for i in range(N):
        rx(2.0 * beta, q[i])

    mz(q)


def qaoa_sample(
    N: int,
    shots: int = 300,
    gammas=(0.3, 0.7, 1.1),
    betas=(0.3, 0.7, 1.1),
):
    best = {
        "best_energy": float("inf"),
        "best_spins": None,
        "best_params": None,
        "best_bitstring": None,
    }

    for gamma in gammas:
        for beta in betas:
            result = cudaq.sample(
                qaoa_kernel,
                N,
                float(gamma),
                float(beta),
                shots_count=shots,
            )

            for bitstring, _freq in result.items():
                spins = bitstring_to_spins(bitstring)
                E = labs_energy(spins)
                if E < best["best_energy"]:
                    best["best_energy"] = E
                    best["best_spins"] = spins
                    best["best_params"] = (float(gamma), float(beta))
                    best["best_bitstring"] = bitstring

    return best


def try_set_target(name: str):
    try:
        cudaq.set_target(name)
        return True
    except Exception as e:
        print(f"[warn] set_target('{name}') failed: {e}")
        return False


if __name__ == "__main__":
    print(f"CUDA-Q version: {getattr(cudaq, '__version__', 'unknown')}")

    # On Brev GPU images this should work
    try_set_target("nvidia")

    # Small-N validation vs brute force
    for N in [3, 4, 5, 6]:
        q = qaoa_sample(N, shots=300)
        bf_E, _ = brute_force_best(N)
        print(f"N={N} | QAOA E={q['best_energy']} | Brute E={bf_E} | params={q['best_params']}")

    # Scale-up sanity check (no brute force)
    for N in [8, 10, 12]:
        q = qaoa_sample(N, shots=500, gammas=(0.2, 0.6, 1.0), betas=(0.2, 0.6, 1.0))
        print(f"N={N} | best found E={q['best_energy']} | params={q['best_params']}")
