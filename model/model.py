import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





df = pd.read_csv("./various_food_data.csv")
print(df.head)
print(df.shape)


# Data Cleaning 
print(df.isnull().sum())

# Removing unwanted columns
df.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1, inplace=True)
print(df.tail)

# Dropping duplicates
df.drop_duplicates()
print(df.shape)

# Identifying outliers using box plot
# plt.figure(figsize(23),3)
# sns.boxplot(data = df[['Zinc']])
# plt.title("Boxplot of selling_price and purchase_price", fontsize=15)
# # plt.show()


df['Fat_Missing'] = df['Fat'].isnull().astype(int)
print(df.head)


# Data Visualization


# Normalizing data
scaler = StandardScaler()

numeric_columns = ['Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats',
    'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3',
    'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K',
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese', 'Phosphorus', 'Potassium',
    'Selenium', 'Zinc', 'Nutrition Density']


df_scaled = scaler.fit_transform(df[numeric_columns])
df_scaled = pd.DataFrame(df_scaled,columns=numeric_columns)

print(df_scaled.head(10))


# feature engineering 

df['Caloric Density'] = df['Caloric Value'] / df['Water']
print(df.info())


print(df.isin([np.inf, -np.inf]).sum())
df['Caloric Density'].replace([np.inf, -np.inf], np.nan).fillna(df.select_dtypes(include='number').mean(),inplace=True)
print(df.head(10))
print(df.isnull().sum())


df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Verify that there are no NaN or infinite values left
print(df.isnull().sum())
print(df.isin([np.inf, -np.inf]).sum())



# Ensure all values are non-negative before applying log transformation
for col in numeric_columns:
    if (df[col] < -1).any():  # Check if any values are less than -1
        print(f"Column {col} contains values less than -1. Skipping log transformation.")
    else:
        df[col] = np.log1p(df[col])


# Step 1: Identify numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Step 2: Calculate Z-scores
z_scores = df[numeric_columns].apply(zscore)

# Step 3: Identify outliers
threshold = 3
outliers = (z_scores.abs() > threshold)

for col in numeric_columns:
    df[col] = np.log1p(df[col])


sns.boxplot(y=df['Caloric Value'])
plt.title('Boxplot of Caloric Value')
# plt.show()

# Remove rows with outliers
df_no_outliers = df[~outliers.any(axis=1)]

df2 = df_no_outliers




# After handling outliers
sns.histplot(df_no_outliers['Caloric Value'], kde=True, label='After', color='orange')
plt.title('Distribution of Caloric Value (After)')
# plt.show()



# MODEL TRAINING 

features = numeric_columns
target = 'Caloric Value'

# Create feature matrix (X) and target vector (y)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Caloric Value')
plt.ylabel('Predicted Caloric Value')
plt.title('Actual vs Predicted Caloric Value')
plt.show()