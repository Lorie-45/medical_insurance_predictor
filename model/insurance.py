import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


df = pd.read_csv("./insurance.csv")

print(df.tail(10))
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)

# Display duplicates
duplicates = df[df.duplicated()]
print(duplicates)

df = df.drop_duplicates()
print(df.shape)


# Data Visualization
sns.histplot(df["charges"], bins=30, kde=True)
plt.title("Insurance Charges Distribution")
# plt.show()


# Encoding

# one hot
df = pd.get_dummies(df, columns=["region"], drop_first=True)


# binary
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})


print(df)
print(df.dtypes)

# Correlation matrix
corr = df.corr()
# sns.heatmap(corr, annot= True, cmap='BuPu')
plt.title("Correlation Matrix")
# plt.show()


# outlier detection
sns.boxplot(df, y="charges")
plt.title("Outlier boxplot")
# plt.show()


# Function to remove outliers
def remove_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


df = remove_outliers("bmi")
df = remove_outliers("charges")

print(df.shape)

# training and testing

X = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))


plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges (After Outlier Removal)")
plt.show()

coefficients = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_}
).sort_values(by="Coefficient", ascending=False)

print(coefficients)
