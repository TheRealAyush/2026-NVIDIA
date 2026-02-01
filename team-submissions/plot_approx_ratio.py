import matplotlib.pyplot as plt

N = [3, 4, 5, 6]
qaoa_E = [1, 2, 2, 7]
optimal_E = [1, 2, 2, 7]

ratio = [q/o for q, o in zip(qaoa_E, optimal_E)]

plt.plot(N, ratio, marker='o')
plt.xlabel("Problem Size (N)")
plt.ylabel("Approximation Ratio")
plt.title("QAOA Approximation Ratio vs N (CPU Validation)")
plt.grid()
plt.savefig("approx_ratio_vs_N.png")
