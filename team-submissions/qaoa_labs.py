import numpy as np
import itertools
import cudaq

# -----------------------------
# LABS energy (classical)
# -----------------------------
def labs_energy(spins):
    N = len(spins)
    E = 0
    for k in range(1, N):
        Ck = sum(spins[i] * spins[i + k] for i in range(N - k))
        E += Ck * Ck
    return E


# -----------------------------
# QAOA kernel (p = 1)
# -----------------------------
@cudaq.kernel
def qaoa_kernel(gamma: float, beta: float, N: int):
    q = cudaq.qvector(N)

    # |+> state
    for i in range(N):
        h(q[i])

    # Cost Hamiltonian (LABS)
    for k in range(1, N):
        for i in range(N - k):
            cx(q[i], q[i + k])
            rz(2.0 * gamma, q[i + k])
            cx(q[i], q[i + k])

    # Mixer
    for i in range(N):
        rx(2.0 * beta, q[i])

    mz(q)


# -----------------------------
# QAOA sampling
# -----------------------------
def qaoa_sample(
    N,
    shots=200,
    gammas=(0.3, 0.7, 1.1),
    betas=(0.3, 0.7, 1.1),
):
    cudaq.set_target("qpp-cpu")

    best_E = float("inf")
    best_spins = None

    for g in gammas:
        for b in betas:
            result = cudaq.sample(
                qaoa_kernel,
                g,
                b,
                N,
                shots_count=shots,
            )

            for bitstring, _ in result.items():
                spins = np.array([1 if c == "0" else -1 for c in bitstring])
                E = labs_energy(spins)
                if E < best_E:
                    best_E = E
                    best_spins = spins

    return best_spins, best_E


# -----------------------------
# Brute force (verification)
# -----------------------------
def brute_force_best(N):
    best_E = float("inf")
    for bits in itertools.product([1, -1], repeat=N):
        best_E = min(best_E, labs_energy(bits))
    return best_E


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    for N in [3, 4, 5, 6]:
        q_spins, q_E = qaoa_sample(N, shots=300)
        bf_E = brute_force_best(N)
        print(f"N={N} | QAOA E={q_E} | Brute E={bf_E}")
