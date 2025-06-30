import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# "whitegrid" for better looking plots
sns.set_style("whitegrid")

# load iris dataset
iris = sns.load_dataset('iris')

print("shape", iris.shape)

print("Column names:", iris.columns.tolist())

print(iris.head())

# scatter plot sepal lenght and sepal width
plt.figure(figsize=(8,6))
sns.scatterplot(data=iris,x="sepal_length", y="sepal_width",hue="species")
plt.title("Scatter Plot of Sepal Length vs Sepal Width")
plt.show()

# histogram of petal lenght
plt.figure(figsize=(8,6))
sns.histplot(iris["petal_length"], kde=True)
plt.title("Histogram")
plt.xlabel("petal_length")
plt.ylabel("Frequency")
plt.show()

# box plot of petal width
plt.figure(figsize=(8,6))
sns.boxplot(data=iris, x="species", y="petal_width")
plt.title("Box plot")
plt.xlabel("species")
plt.ylabel("petal_Width")
plt.show()


# pair plant of entire dataset
sns.pairplot(iris, hue="species")
plt.suptitle("Pairwise Relationships in Iris Dataset", y=1.02)
plt.show()