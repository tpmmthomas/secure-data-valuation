from matplotlib.ticker import ScalarFormatter

# import random
# import os
# N = 10000  # Number of random values to generate per player

# # Generate random values for each player
# player1_values = [f"{random.random():.6f}" for _ in range(N)]
# player2_values = [f"{random.random():.6f}" for _ in range(N)]
# p0_path = os.path.join('../MP-SPDZ/Player-Data','Input-P0-0')
# p1_path = os.path.join('../MP-SPDZ/Player-Data','Input-P1-0')

# print(p0_path)
# # Write the values to text files (one line per file with space-separated values)
# with open(p0_path, "w") as f1:
#     f1.write(" ".join(player1_values))

# with open(p1_path, "w") as f2:
#     f2.write(" ".join(player2_values))

# print("hello")


import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Data: vector dimensions and corresponding computation times
dimensions = [10, 100, 1000, 10000]
times = [0.559696, 1.14082, 6.70172, 63.5648]

plt.figure(figsize=(8, 6))
plt.plot(dimensions, times, marker='o', linestyle='-', color='b')

plt.xlabel('Dimension of Vectors', fontsize=16)
plt.ylabel('Time (seconds)', fontsize=16)
plt.title('Euclidean Distance Computation Time', fontsize=20)

plt.xscale('log')
plt.yscale('log')

ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

# Manually specify y-axis ticks to ensure the maximum (~63) appears in the labels.
# Here we include the lower bound, a couple of intermediate values, and 63.
ax.set_yticks([0.5, 1, 6.70172, 63])
# plt.ylim(0.5, 63)  # Set the y-axis range to clearly show the maximum

plt.grid(True, which="both", ls="--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('results/euclidean_distance_time.png', dpi=300)
plt.show()
