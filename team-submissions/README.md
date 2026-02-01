# Quantum-Enhanced Optimization with GPU Acceleration
*NVIDIA iQuHACK 2026 · LABS Challenge*

---

**Participant:** Ayush  
**Team:** NVIDIAyush  
**Challenge Track:** NVIDIA LABS  
**Hardware:** NVIDIA L4 GPU (Brev)  
**Frameworks:** CUDA-Q · CuPy · NumPy

---

## Project Overview

This project addresses the Low Autocorrelation Binary Sequence (LABS) optimization problem using a hybrid quantum–classical workflow. A QAOA-based quantum routine is combined with GPU-accelerated classical optimization to explore both correctness and performance scaling.

The work emphasizes:
- Correctness before acceleration
- Rigorous verification
- GPU acceleration of both quantum simulation and classical search
- Clear benchmarking and visualization

---

## Repository Structure

team-submissions/

- qaoa_labs.py  
  QAOA implementation using CUDA-Q

- classical_gpu.py  
  GPU-accelerated LABS energy evaluation using CuPy

- gpu_local_search.py  
  GPU-accelerated local search (MTS-style)

- tests.py  
  Automated verification test suite

- TEST_SUITE.md  
  Explanation of verification strategy and test coverage

- plot_bench.py  
  Runtime and speedup plotting

- plot_approx_ratio.py  
  Approximation ratio vs problem size

- classical_gpu_bench.csv  
  CPU vs GPU benchmark data

- time_vs_N_B4096.png  
- speedup_vs_N_B4096.png  
- speedup_vs_B_N100.png  
- approx_ratio_vs_N.png  

- PRD.md  
  Product Requirements Document

- AI_REPORT.md  
  AI usage, verification strategy, and failure analysis

- NVIDIA_IQuHACK_2026_LABS_Ayush.pdf  
  Slide deck

- NVIDIA_IQuHACK_2026_LABS_Ayush.mp4  
  Voiceover for slide deck

- README.md  
  This file

---

## Phase 1 – CPU Validation (Correctness)

### What Was Implemented

- p = 1 QAOA for the LABS Hamiltonian using CUDA-Q
- Brute-force comparison for small problem sizes (N = 3–6)
- Verification of physical and mathematical invariants

### Validation Criteria

- QAOA energy matches brute-force optimum for small N
- Spin-flip symmetry: E(s) = E(-s)
- Energies are non-negative integers

### How to Run

CPU validation and tests:

python qaoa_labs.py  
python tests.py  

Expected behavior:
- QAOA and brute-force energies match for small N
- All verification tests pass

---

## Phase 2 – GPU Acceleration

### Step A: Quantum Simulation on GPU

- Migrated CUDA-Q target from CPU to NVIDIA GPU backend
- Used cuStateVec-based state-vector simulation
- Verified identical energies before and after migration

### Step B: Classical GPU Acceleration

- Replaced NumPy with CuPy for batch energy evaluation
- Accelerated inner loops of classical optimization
- Benchmarked CPU vs GPU performance

### Step C: GPU-Accelerated Local Search

- Implemented GPU-accelerated local search (MTS-style)
- Compared CPU vs GPU runtime for identical configurations
- Observed speedups exceeding 200× for large batch sizes

---

## Running GPU Benchmarks

Verify GPU environment:

nvidia-smi  
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

Classical GPU benchmark:

python classical_gpu.py

GPU local search comparison:

python gpu_local_search.py

Generate plots:

python plot_bench.py  
python plot_approx_ratio.py  

Generated figures include:
- Time vs N (CPU vs GPU)
- Speedup vs N
- Speedup vs batch size
- QAOA approximation ratio vs N

---

## Verification Strategy

All verification logic is implemented in tests.py.

Test coverage includes:
- LABS energy matches reference implementation
- Spin-flip symmetry validation
- Energy is non-negative integer
- Brute-force consistency for small N
- QAOA smoke test

Run tests:

python tests.py

Expected output includes only passing checks.

Additional details are documented in TEST_SUITE.md.

---

## Results Summary

- Correctness fully validated against brute-force solutions
- Quantum GPU acceleration via CUDA-Q backend
- Classical GPU acceleration via CuPy with large speedups
- Successful scaling to problem sizes N ≥ 150
- All plots generated programmatically

---

## AI Usage & Verification

AI assistance was used for:
- Prototyping CUDA-Q kernels
- Structuring GPU acceleration strategies
- Debugging kernel compilation issues

All AI-generated code was explicitly verified through:
- Unit tests
- Cross-checks with brute-force results
- Physical invariants

Full documentation is provided in AI_REPORT.md.

---

## Final Notes

This project demonstrates a rigorous, end-to-end GPU-accelerated quantum–classical workflow with strong emphasis on correctness, verification, and performance analysis. The implementation is modular, reproducible, and extensible to larger problem sizes and multi-GPU configurations.
