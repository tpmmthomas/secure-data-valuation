import re
import matplotlib.pyplot as plt

all_acc = []
for seed in range(1, 6):
    # Path to your txt file (update the path if needed)
    filename = f'results/exp2_cifar10_resnet18_{seed}.txt'

    # Dictionary to store method names and their corresponding accuracy sequences
    accuracy_data = {}

    # Open and read the file
    with open(filename, 'r') as f:
        for line in f:
            # Look for lines that contain an Accuracy sequence
            if "Accuracy sequence:" in line:
                # Extract the method name (text before " Accuracy sequence:")
                method = line.split(" Accuracy sequence:")[0].strip()
                
                # Use a regex to extract the list content between [ and ]
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    # Convert the comma-separated string into a list of floats
                    accuracy_list = [float(x.strip()) for x in match.group(1).split(',')]
                    accuracy_data[method] = accuracy_list
    all_acc.append(accuracy_data)


# Plotting the accuracy sequences
plt.figure(figsize=(10, 6))

# Matplotlib's default color cycle is used to ensure different colors for each line
for method, accuracies in all_acc[0].items():
    #Compute mean of all accuracies
    mean_acc = [sum([acc[method][i] for acc in all_acc])/5 for i in range(len(accuracies))]
    plt.plot(range(len(accuracies)), mean_acc, marker='o', label=method)
    # plt.plot(range(len(accuracies)), accuracies, marker='o', label=method)

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('CIFAR10 Resnet18 Accuracy Curve for Different AL Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('results/exp2_cifar10_resnet18.png', dpi=300)
plt.show()
