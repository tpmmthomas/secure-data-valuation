import math

def calculate_mean_sd(values):
    """
    Calculate the mean and standard deviation of a list of values.

    Parameters:
        values (list of float): The list of numerical values.

    Returns:
        tuple: A tuple (mean, standard_deviation).
    """
    if not values:
        raise ValueError("The input list cannot be empty.")

    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    standard_deviation = math.sqrt(sum(squared_diffs) / len(values))
    return mean, standard_deviation

# Example usage:
if __name__ == "__main__":
    sample_values = [0.4947894427836657, 0.4904365193948503, 0.36706789879196344, 0.498195076768044, 0.5508109616389822]
    mean, sd = calculate_mean_sd(sample_values)
    print("Mean:", mean)
    print("Standard Deviation:", sd)