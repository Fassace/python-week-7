import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Task 1: Load and Explore the Dataset ---
# Load the Iris dataset from a URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
try:
    df = pd.read_csv(url, header=None)
    # Rename columns based on the dataset description
    df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Display the first few rows of the dataset
print("First five rows of the dataset:")
print(df.head())

# Check the dataset structure and any missing values
print("\nDataset information:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values with column means if necessary
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(numeric_only=True), inplace=True)

# --- Task 2: Basic Data Analysis ---
# Compute basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Group by species and compute the mean of numerical columns
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Find maximum mean value for each species
max_mean = grouped.max(axis=1)
print("\nMaximum mean values per species:")
print(max_mean)

# --- Task 3: Data Visualization ---
# 1. Bar Chart (Comparing numerical values across categories: species vs sepal length)
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# 2. Histogram (Distribution of a numerical column: sepal length)
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal length (cm)'], kde=True, color='blue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 3. Scatter Plot (Relationship between two numerical columns: sepal length vs petal length)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='deep')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
