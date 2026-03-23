# credit-card-fraud-detection-
Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/credit_card_fraud_dataset.csv")

# Basic info
print(df.head())
print(df.info())

# Convert date column
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# -------------------------------
# 1. Fraud vs Non-Fraud Count
# -------------------------------
plt.figure()
sns.countplot(x='IsFraud', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Is Fraud (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 2. Transaction Amount Distribution
# -------------------------------
plt.figure()
sns.histplot(df[df['IsFraud'] == 0]['Amount'], bins=50, kde=True, label="Non-Fraud")
sns.histplot(df[df['IsFraud'] == 1]['Amount'], bins=50, kde=True, color='red', label="Fraud")
plt.legend()
plt.title("Transaction Amount Distribution")
plt.show()

# -------------------------------
# 3. Fraud by Transaction Type
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='TransactionType', hue='IsFraud', data=df)
plt.title("Fraud by Transaction Type")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 4. Top Locations with Fraud
# -------------------------------
fraud_locations = df[df['IsFraud'] == 1]['Location'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=fraud_locations.index, y=fraud_locations.values)
plt.title("Top 10 Fraud Locations")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 5. Time-based Analysis
# -------------------------------
df['Date'] = df['TransactionDate'].dt.date
fraud_by_date = df[df['IsFraud'] == 1].groupby('Date').size()

plt.figure(figsize=(12,5))
fraud_by_date.plot()
plt.title("Fraud Transactions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Frauds")
plt.show()

# -------------------------------
# 6. Correlation Heatmap
# -------------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
from sklearn.utils import resample

# Separate classes
df_majority = df[df.IsFraud == 0]
df_minority = df[df.IsFraud == 1]

# Upsample fraud cases
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

# Combine
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle
df_balanced = df_balanced.sample(frac=1)

# Use this new dataset
X = df_balanced.drop('IsFraud', axis=1)
y = df_balanced['IsFraud']



