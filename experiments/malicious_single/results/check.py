import math

# Define the list x
x = [153131416.0, 153131416.0, 153131416.0, 153131416.0, 153131416.0]

# Calculate mean
mean = sum(x) / len(x)

# Calculate standard deviation (population standard deviation)
variance = sum((xi - mean) ** 2 for xi in x) / len(x)
sd = math.sqrt(variance)

print("Mean:", mean, "Standard Deviation:", sd)