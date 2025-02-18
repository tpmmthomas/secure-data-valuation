import scipy.stats as stats
import pandas as pd

df = pd.read_csv("results/experiment_1_cifar10_resnet18.csv")

random = df['Random'].values
shapley = df['Shapley'].values
entropy = df['Entropy'].values
loss = df['Loss'].values


# Calculate the Spearman rank correlation coefficient and the p-value

print("CIFAR10 / Resnet18 Spearman Rank Correlation Coefficient")
corr, p_value = stats.spearmanr(random, shapley)
print("Random/Shapley:", corr)
corr, p_value = stats.spearmanr(entropy, shapley)
print("Entropy/Shapley:", corr)
corr, p_value = stats.spearmanr(loss, shapley)
print("Loss/Shapley:", corr)
