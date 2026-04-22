import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("python dataset.csv")

# Global Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14,8)

# Bar Chart
df['Species'].value_counts().plot(
    kind='bar', 
    color=sns.color_palette("viridis"),
    legend=False
)

plt.title("Species Distribution", fontsize=16)
plt.xlabel("Species", labelpad=10)   
plt.ylabel("Count")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# Count Plot
sns.countplot(
    x='Migration_Success',
    data=df,
    hue='Migration_Success',
    palette='Set2',
    legend=False
)
plt.title("Migration Success Count", fontsize=16)
plt.show()

# Scatter Plot
plt.figure(figsize=(14,8))
plt.scatter(
    df['Start_Latitude'], 
    df['Start_Longitude'],
    c=df['Tag_Weight_g'],   # color based on weight 
    cmap='coolwarm',
    alpha=0.5,
    s=20
)
plt.colorbar(label="Tag Weight")
plt.title("Start Location Distribution", fontsize=16)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

# Histogram
sns.histplot(
    df['Tag_Weight_g'], 
    bins=30, 
    kde=True, 
    color='purple'
)
plt.title("Tag Weight Distribution", fontsize=16)
plt.show()

# Boxplot

plt.title("Tag Weight vs Migration Success", fontsize=16)
sns.boxplot(
    x='Migration_Success',
    y='Tag_Weight_g',
    data=df,
    hue='Migration_Success',
    palette='cool',
    legend=False
)
plt.show()

# Heatmap (Clean + Colorful)
important_cols = ['Tag_Weight_g', 'Recovery_Time_days', 'Observation_Counts']
corr_subset = df[important_cols].corr()

plt.figure(figsize=(10,6))
sns.heatmap(
    corr_subset, 
    annot=True, 
    cmap='Spectral',
    linewidths=1
)
plt.title("Important Correlation Heatmap", fontsize=16)
plt.show()
#####UNIT IV: EDA

# Summary Statistics
df.describe()

# Missing Values
df.isnull().sum()

# Correlation
df.corr(numeric_only=True)

# Covariance
df.cov(numeric_only=True)

# Outlier Detection (Tag Weight)
Q1 = df['Tag_Weight_g'].quantile(0.25)
Q3 = df['Tag_Weight_g'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Tag_Weight_g'] < Q1 - 1.5*IQR) | (df['Tag_Weight_g'] > Q3 + 1.5*IQR)]
outliers
print(df.describe())

print(df.isnull().sum())

print(df.corr(numeric_only=True))

print(df.cov(numeric_only=True))

print("Number of outliers:", len(outliers))
#print(outliers.head())


######UNIT V: Statistical Analysis
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.weightstats import ztest

# Chi-Square Test
table = pd.crosstab(df['Region'], df['Migration_Success'])
chi2, p, dof, expected = stats.chi2_contingency(table)



# VIF
numeric_df = df[['Tag_Weight_g', 'Recovery_Time_days', 'Observation_Counts']].dropna()
X = sm.add_constant(numeric_df)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_data

# Normal Distribution (Tag Weight)
plt.figure(figsize=(14,7))

sns.histplot(
    df['Tag_Weight_g'],
    bins=30,
    kde=True,
    color='purple'   # main color
)

plt.title("Normal Distribution of Tag Weight", fontsize=16)
plt.xlabel("Tag Weight (g)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Poisson Distribution (Observation Counts)
plt.figure(figsize=(14,7))

sns.histplot(
    df['Observation_Counts'],
    bins=30,
    color='teal'   #  different color
)

plt.title("Distribution of Observation Counts", fontsize=16)
plt.xlabel("Observation Counts")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
print("Chi-square value:", chi2)
print("Chi-square p-value:", p)

print("\nVIF Data:")
print(vif_data)

#Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Select features
X = df[['Flight_Duration_hours', 'Average_Speed_kmph']]
y = df['Flight_Distance_km']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))
































