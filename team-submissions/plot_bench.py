import csv
from collections import defaultdict

import matplotlib.pyplot as plt

CSV_PATH = "classical_gpu_bench.csv"
OUT1 = "time_vs_N_B4096.png"
OUT2 = "speedup_vs_N_B4096.png"
OUT3 = "speedup_vs_B_N100.png"

def read_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "N": int(float(r["N"])),
                "B": int(float(r["B"])),
                "t_cpu_s": float(r["t_cpu_s"]),
                "t_gpu_s": float(r["t_gpu_s"]),
                "speedup": float(r["speedup"]),
            })
    return rows

def main():
    rows = read_rows(CSV_PATH)

    # -------- Plot 1 + 2: fixed B=4096 vs N --------
    B_fixed = 4096
    rB = sorted([r for r in rows if r["B"] == B_fixed], key=lambda x: x["N"])
    Ns = [r["N"] for r in rB]
    cpu = [r["t_cpu_s"] for r in rB]
    gpu = [r["t_gpu_s"] for r in rB]
    spd = [r["speedup"] for r in rB]

    plt.figure()
    plt.plot(Ns, cpu, marker="o", label="CPU")
    plt.plot(Ns, gpu, marker="o", label="GPU")
    plt.xlabel("N")
    plt.ylabel("Time (s)")
    plt.title(f"Time vs N (batch size B={B_fixed})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT1, dpi=200)

    plt.figure()
    plt.plot(Ns, spd, marker="o")
    plt.xlabel("N")
    plt.ylabel("Speedup (CPU/GPU)")
    plt.title(f"Speedup vs N (batch size B={B_fixed})")
    plt.tight_layout()
    plt.savefig(OUT2, dpi=200)

    # -------- Plot 3: fixed N=100 vs B --------
    N_fixed = 100
    rN = sorted([r for r in rows if r["N"] == N_fixed], key=lambda x: x["B"])
    Bs = [r["B"] for r in rN]
    spdB = [r["speedup"] for r in rN]

    plt.figure()
    plt.plot(Bs, spdB, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size B (log scale)")
    plt.ylabel("Speedup (CPU/GPU)")
    plt.title(f"Speedup vs Batch Size (N={N_fixed})")
    plt.tight_layout()
    plt.savefig(OUT3, dpi=200)

    print("Saved plots:")
    print(" -", OUT1)
    print(" -", OUT2)
    print(" -", OUT3)

if __name__ == "__main__":
    main()
