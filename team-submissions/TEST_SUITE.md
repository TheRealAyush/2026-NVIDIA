# Test Suite and Verification Strategy

This submission verifies correctness at three layers: (1) the LABS energy function, (2) brute-force ground truth for small N, and (3) the quantum-enhanced sampling workflow.

## What is being verified

### 1) Energy correctness (core physics / objective)
- `labs_energy(spins)` is tested against an independent reference implementation on randomized spin vectors for N=3..12.
- This ensures the objective being optimized is correct.

### 2) Physical invariances / sanity checks
- Spin-flip symmetry: `E(s) == E(-s)` is validated for many randomized inputs across N=3..24.
- Nonnegativity and integrality: energy is always a nonnegative integer because it is a sum of squared integer correlations.

### 3) Brute-force consistency (ground truth for small N)
- `brute_force_best(N)` is checked to ensure:
  - Its reported energy equals `labs_energy(best_spins)`.
  - Its reported best energy is not worse than many random samples (sanity for minimality).

### 4) QAOA workflow (smoke test)
- `qaoa_sample(N, shots=...)` is executed for N in {3,4,5,6} with low shot count.
- The test confirms:
  - Output spins have correct length and are ±1.
  - Reported `best_energy` matches `labs_energy(best_spins)`.

## How to run tests

From the `team-submissions` directory:
```bash
python tests.py
