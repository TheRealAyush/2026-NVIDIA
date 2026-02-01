"""
tests.py

Run:
  python tests.py

What this does:
- Verifies LABS energy matches a small reference implementation
- Checks physical invariances (spin-flip symmetry)
- Confirms brute-force best is consistent
- Smoke-tests QAOA sampling for small N (fast)
- Prints a clear PASS/FAIL summary

Notes:
- These tests are designed to be quick.
- If CUDA-Q is not available in the environment, QAOA tests will be skipped.
"""

import math
import random
import traceback

# ----------------------------
# Reference LABS energy
# ----------------------------
def ref_labs_energy(spins):
    # spins is a list of +/-1
    N = len(spins)
    E = 0
    for k in range(1, N):
        ck = 0
        for i in range(N - k):
            ck += spins[i] * spins[i + k]
        E += ck * ck
    return E


def negate(spins):
    return [-s for s in spins]


# ----------------------------
# Tiny test framework
# ----------------------------
class TestFailure(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise TestFailure(msg)


def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except TestFailure as e:
        print(f"[FAIL] {name}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        traceback.print_exc()
        return False


# ----------------------------
# Import student code
# ----------------------------
def import_student():
    try:
        import qaoa_labs as student
        return student
    except Exception as e:
        raise TestFailure(f"Could not import qaoa_labs.py: {e}")


# ----------------------------
# Tests
# ----------------------------
def test_energy_matches_reference():
    student = import_student()
    check(hasattr(student, "labs_energy"), "qaoa_labs.py must define labs_energy(spins)")
    labs_energy = student.labs_energy

    rng = random.Random(0)
    for N in range(3, 13):
        for _ in range(25):
            spins = [rng.choice([-1, 1]) for _ in range(N)]
            e_ref = ref_labs_energy(spins)
            e_stu = labs_energy(spins)
            check(e_stu == e_ref, f"Energy mismatch for N={N}: got {e_stu}, expected {e_ref}")


def test_energy_spin_flip_symmetry():
    student = import_student()
    labs_energy = student.labs_energy

    rng = random.Random(1)
    for N in range(3, 25):
        spins = [rng.choice([-1, 1]) for _ in range(N)]
        e1 = labs_energy(spins)
        e2 = labs_energy(negate(spins))
        check(e1 == e2, f"Spin-flip symmetry violated for N={N}: E(s)={e1}, E(-s)={e2}")


def test_energy_nonnegative_and_integer():
    student = import_student()
    labs_energy = student.labs_energy

    rng = random.Random(2)
    for N in range(3, 30):
        spins = [rng.choice([-1, 1]) for _ in range(N)]
        e = labs_energy(spins)
        check(isinstance(e, int), f"Energy should be int, got type {type(e)}")
        check(e >= 0, f"Energy should be nonnegative, got {e}")


def test_bruteforce_consistency_smallN():
    student = import_student()
    check(hasattr(student, "brute_force_best"), "qaoa_labs.py must define brute_force_best(N)")
    brute_force_best = student.brute_force_best
    labs_energy = student.labs_energy

    for N in range(3, 11):
        best_e, best_spins = brute_force_best(N)
        check(best_e == labs_energy(best_spins), f"brute_force_best energy mismatch for N={N}")
        # sanity: cannot be worse than any random sample
        rng = random.Random(N)
        for _ in range(50):
            spins = [rng.choice([-1, 1]) for _ in range(N)]
            check(best_e <= labs_energy(spins), f"brute_force_best not minimal for N={N}")


def test_qaoa_smoketest_smallN_fast():
    # This is intentionally a smoke test: it checks "runs and returns valid data".
    student = import_student()
    check(hasattr(student, "qaoa_sample"), "qaoa_labs.py must define qaoa_sample(N, ...)")
    qaoa_sample = student.qaoa_sample
    labs_energy = student.labs_energy

    # If CUDA-Q is missing, skip cleanly.
    try:
        import cudaq  # noqa: F401
    except Exception:
        print("[SKIP] QAOA smoke test: cudaq not available in this environment")
        return

    # Keep shots low so this runs fast everywhere.
    for N in [3, 4, 5, 6]:
        res = qaoa_sample(N, shots=200)
        # Accept either dict return or tuple return, depending on your implementation.
        if isinstance(res, dict):
            check("best_spins" in res and "best_energy" in res, "qaoa_sample dict must include best_spins and best_energy")
            best_spins = res["best_spins"]
            best_e = res["best_energy"]
        elif isinstance(res, tuple) and len(res) >= 2:
            best_spins, best_e = res[0], res[1]
        else:
            raise TestFailure("qaoa_sample must return dict or tuple (best_spins, best_energy, ...)")

        check(len(best_spins) == N, f"best_spins length must be N={N}")
        check(all(s in [-1, 1] for s in best_spins), "best_spins must be +/-1 spins")
        check(best_e == labs_energy(best_spins), "Reported best_energy must match labs_energy(best_spins)")


def main():
    tests = [
        ("LABS energy matches reference", test_energy_matches_reference),
        ("Spin-flip symmetry E(s)=E(-s)", test_energy_spin_flip_symmetry),
        ("Energy is nonnegative integer", test_energy_nonnegative_and_integer),
        ("Brute force best is consistent (small N)", test_bruteforce_consistency_smallN),
        ("QAOA smoke test (small N)", test_qaoa_smoketest_smallN_fast),
    ]

    ok = 0
    for name, fn in tests:
        if run_test(name, fn):
            ok += 1

    total = len(tests)
    print(f"\nSummary: {ok}/{total} tests passed")
    if ok != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
