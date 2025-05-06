import matplotlib.pyplot as plt

# Hardcoded scalability test results
N_values = [500, 1000, 5000, 10000]
N_times = [120.96316504478455, 125.67967057228088, 124.46738600730896, 120.83730101585388]

M_values = [10, 20, 50, 100]
M_times = [124.13601016998291, 139.49904990196228, 183.29345107078552, 247.79890704154968]

DIM_values = [20, 50, 100, 500]
DIM_times = [121.4334409236908, 138.5918505191803, 173.93087196350098, 436.055855512619]

K_values = [10, 20, 50, 100]
K_times = [106.06979966163635, 143.54053330421448, 247.9976189136505, 433.11687207221985]

# Compute global y-axis limits
all_times = N_times + M_times + DIM_times + K_times
ymin, ymax = min(all_times), max(all_times)

# Create a 1x4 grid of subplots with shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)


# Plot for M (only set 2)
axes[0].plot(M_values, M_times, marker='o')
axes[0].set_title('Runtime vs size of I')
axes[0].set_xlabel('size of I\n(a)')
axes[0].set_ylim(ymin, ymax)

# Plot for DIM
axes[1].plot(DIM_values, DIM_times, marker='o')
axes[1].set_title('Runtime vs Dimension')
axes[1].set_xlabel('Dimension\n(b)')
axes[1].set_ylim(ymin, ymax)

# Plot for K
axes[2].plot(K_values, K_times, marker='o')
axes[2].set_title('Runtime vs size of D_R')
axes[2].set_xlabel('size of D_R\n(c)')
axes[2].set_ylim(ymin, ymax)

plt.tight_layout()
for ax in axes:
    # Set title and label font sizes
    ax.title.set_fontsize(18)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    # Set tick label font sizes
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)
fig.savefig('scalability_results.png', bbox_inches='tight')
