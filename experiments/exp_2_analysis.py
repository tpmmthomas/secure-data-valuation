import re
import matplotlib.pyplot as plt

# Path to your txt file (update the path if needed)
filename = 'results/exp2_mnist_cnn.txt'

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

# Plotting the accuracy sequences
plt.figure(figsize=(10, 6))

# Matplotlib's default color cycle is used to ensure different colors for each line
for method, accuracies in accuracy_data.items():
    plt.plot(range(len(accuracies)), accuracies, marker='o', label=method)

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('MNIST / CNN Accuracy Curve for Different AL Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('results/exp2_mnist_cnn.png', dpi=300)
plt.show()
