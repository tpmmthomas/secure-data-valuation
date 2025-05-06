import math

# Define the list x
x = [1.0264930725097656, 1.007183313369751, 0.9982633590698242, 0.9773850440979004, 1.0405354499816895]

# Calculate mean
mean = sum(x) / len(x)

# Calculate standard deviation (population standard deviation)
variance = sum((xi - mean) ** 2 for xi in x) / len(x)
sd = math.sqrt(variance)

print("Mean:", mean)
print("Standard Deviation:", sd)