from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt #added for plotting

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# you can save the boxplot...
boxplot = pd.plotting.boxplot(df["MedInc"], vert=False)
plt.title("Median Income Boxplot (In Hundreds of Thousands of Dollars)")
plt.xlabel("Median Income (In Hundreds of Thousands of Dollars)")
plt.savefig("figs/boxplot.png")  # Save the boxplot as an image file
plt.show()
plt.close()