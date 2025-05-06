import re
import matplotlib.pyplot as plt

all_acc = []
DATASET = "mnist"
MODEL = "cnn"
NAME = "formal1"

# Load the accuracy data from text files for each seed
for seed in range(21, 26):
    filename = f'results/exp2_{DATASET}_{MODEL}_{NAME}_{seed}.txt'
    accuracy_data = {}
    with open(filename, 'r') as f:
        for line in f:
            if "Accuracy sequence:" in line:
                method = line.split(" Accuracy sequence:")[0].strip()
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    accuracy_list = [float(x.strip()) for x in match.group(1).split(',')]
                    accuracy_data[method] = accuracy_list
    all_acc.append(accuracy_data)

# Create the plot with a larger figure size for clarity
plt.figure(figsize=(12, 8))

# Plot the mean accuracy curves for each method with increased line width and marker size
for method, accuracies in all_acc[0].items():
    mean_acc = [sum([acc[method][i] for acc in all_acc]) / len(all_acc) for i in range(len(accuracies))]
    plt.plot(range(len(accuracies)), mean_acc, marker='o', linewidth=2, markersize=8, label=method)

# Set larger font sizes for the axes labels, title, and tick labels
plt.xlabel('Iteration', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.title('MNIST: Accuracy Curve against different AL methods', fontsize=27)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout()

# Save and display the plot
plt.savefig(f'results/exp2_{DATASET}_{MODEL}_{NAME}.png', dpi=300)
plt.show()
