
# Step 2: Exploratory Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# 3. Check data types and missing values
print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

# 4. Check duplicate records
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# 5. Check the target column
print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

# 6. Summary statistics
print("\nNumerical summary:")
print(df.describe())

print("\nCategorical summary:")
print(df.describe(include="object"))

# 7. Prepare a numeric version of TotalCharges for analysis
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing values after converting TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# 8. Visualize churn distribution
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 9. Compare churn by contract type
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 10. Compare monthly charges
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")
