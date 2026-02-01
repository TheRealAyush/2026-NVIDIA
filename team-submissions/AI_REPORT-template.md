> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have employed AI agents in a thoughtful way.

**Required Sections:**

1. **The Workflow:** How did you organize your AI agents? (e.g., "We used a Cursor agent for coding and a separate ChatGPT instance for documentation").
We used AI as an engineering assistant rather than an autonomous code generator. The workflow was iterative and human-in-the-loop throughout the project.

ChatGPT was used primarily for:
Translating high-level ideas (QAOA structure, LABS cost function, GPU acceleration strategy) into concrete Python and CUDA-Q implementations.
Debugging compiler errors from CUDA-Q by interpreting error messages and suggesting compatible kernel structures.
Structuring verification logic and test coverage aligned with physical invariants.
Helping design benchmarking and plotting scripts for CPU vs GPU performance comparison.

The human developer (us) retained control over:
Algorithmic decisions (choice of QAOA p=1, parameter grids, success metrics).
Performance scaling decisions (which parts to GPU-accelerate and why).
Final validation of correctness by comparing against brute-force baselines and physical constraints.
Deciding when AI suggestions were incorrect or incompatible with CUDA-Q and rewriting them manually.

AI outputs were never accepted blindly; all AI-generated code was either tested immediately or modified based on observed runtime behavior.

2. **Verification Strategy:** How did you validate code created by AI?
All AI-generated code was validated using a combination of unit tests, physical invariants, and cross-checks against known baselines. The goal was explicitly to catch hallucinations or incorrect assumptions introduced by AI.

The following tests are implemented in tests.py:

**LABS energy reference check:**

Verifies that the energy computed by labs_energy() matches a trusted reference implementation for known spin configurations.
    This catches algebraic or indexing errors in AI-generated cost functions.

**Spin-flip symmetry:**

Tests that E(s) == E(-s) for all tested configurations.
    This directly enforces a physical invariant of the LABS Hamiltonian and detects incorrect sign handling.

**Energy non-negativity and integer-valued check:**

Confirms that computed energies are non-negative integers.
    This catches AI hallucinations where floating-point operations or incorrect accumulation were introduced.

**Brute-force consistency (small N):**

For small problem sizes, verifies that brute-force search returns the same minimum energy as the QAOA-based approach when the QAOA solution is optimal.
    This prevents silent logical errors in the optimization loop.

**QAOA smoke test:**

Runs QAOA end-to-end for small N to ensure the kernel compiles, executes, and returns valid results.
    This was critical for catching CUDA-Q kernel incompatibilities introduced by AI suggestions.

All tests pass successfully, and the test suite is designed specifically to catch subtle but realistic AI-generated logic errors.


3. **The "Vibe" Log:**
* *Win:*

  AI dramatically accelerated development during the CUDA-Q debugging phase. CUDA-Q kernel errors can be cryptic, and AI helped interpret compiler error messages and suggest compatible kernel signatures, argument annotations, and control-flow restrictions. This saved hours that would otherwise have been spent searching documentation or experimenting blindly.
* *Learn:*

    Early prompts were too high-level (e.g., “write a QAOA kernel in CUDA-Q”), which often produced code incompatible with CUDA-Q’s strict kernel constraints. We learned to:

  Provide exact error messages.

  Explicitly state which CUDA-Q features were unsupported.

  Ask for minimal, incremental changes instead of full rewrites.

This led to significantly higher-quality responses and fewer hallucinations.
* *Fail:*

    AI initially suggested using gate functions (e.g., h, rz, rzz) that are not available in the CUDA-Q Python API in the expected form. This caused repeated compiler errors.

    We identified the issue by:

  Inspecting available symbols in the cudaq module.

  Rewriting the kernel to use supported constructs and validated patterns.

  Adding tests to ensure kernels compiled and executed successfully.

This failure reinforced the importance of verifying AI suggestions against real compiler behavior rather than assuming API availability.
* *Context Dump:*

    Effective prompts typically included:

  The exact CUDA-Q compiler error output.

  The current kernel code.

  A clear constraint such as: “Do not use unsupported gates; only suggest patterns known to work in CUDA-Q kernels.”

We did not rely on a persistent skills.md file, but instead treated each interaction as a constrained debugging session with strong contextual grounding.

4. **Summary:**

AI was used as a productivity amplifier, not a source of truth. Every AI-generated component was validated through unit tests, physical invariants, or direct comparison to brute-force results. This disciplined approach ensured correctness, avoided silent failures, and aligned with the project’s emphasis on rigorous engineering over raw speed.
