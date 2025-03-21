import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_rows = 3000

# Expanded food lists
foods = {
    "Fruit": ["Apple", "Banana", "Orange", "Grapes", "Strawberries", "Mango", "Pineapple"],
    "Protein": ["Chicken Breast", "Salmon", "Beef Steak", "Tofu", "Eggs", "Shrimp", "Turkey"],
    "Vegetable": ["Broccoli", "Spinach", "Carrot", "Kale", "Bell Pepper", "Cucumber", "Zucchini"],
    "Grain": ["White Rice", "Brown Rice", "Quinoa", "Oatmeal", "Whole Wheat Bread", "Pasta"],
    "Dairy": ["Milk", "Yogurt", "Cheese", "Cottage Cheese", "Butter"],
    "Nut": ["Almonds", "Walnuts", "Peanuts", "Cashews", "Pistachios"],
    "Snack": ["Chips", "Chocolate Bar", "Cookies", "Granola Bar", "Popcorn"]
}

# Generate food names and categories
food_data = []
categories = []
for _ in range(num_rows):
    category = np.random.choice(list(foods.keys()))
    food_name = np.random.choice(foods[category])
    food_data.append(food_name)
    categories.append(category)

# Calculate nutrient values before creating the DataFrame
carbs = np.round(np.where(
    np.isin(categories, ["Fruit", "Grain", "Snack"]),
    np.random.uniform(15, 80, size=num_rows),  # Higher carbs for grains/fruits
    np.random.uniform(0, 30, size=num_rows)
), 1)

protein = np.round(np.where(
    np.isin(categories, ["Protein", "Nut", "Dairy"]),
    np.random.uniform(10, 40, size=num_rows),  # Higher protein for meats/nuts
    np.random.uniform(0, 15, size=num_rows)
), 1)

fat = np.round(np.where(
    np.isin(categories, ["Nut", "Dairy", "Snack"]),
    np.random.uniform(10, 45, size=num_rows),  # Higher fat for nuts/snacks
    np.random.uniform(0, 20, size=num_rows)
), 1)

sugar = np.round(np.where(
    np.isin(categories, ["Fruit", "Snack"]),
    np.random.uniform(10, 50, size=num_rows),  # Higher sugar for fruits/snacks
    np.random.uniform(0, 15, size=num_rows)
), 1)

fiber = np.round(np.where(
    np.isin(categories, ["Vegetable", "Grain"]),
    np.random.uniform(3, 12, size=num_rows),  # Higher fiber for veggies/grains
    np.random.uniform(0, 5, size=num_rows)
), 1)

cholesterol = np.where(
    np.isin(categories, ["Protein", "Dairy"]),
    np.random.randint(30, 150, size=num_rows),
    np.random.randint(0, 50, size=num_rows)
)

iron = np.round(np.where(
    np.isin(categories, ["Vegetable"]),
    np.random.uniform(1, 5, size=num_rows),
    np.random.uniform(0, 3, size=num_rows)
), 1)

# Create DataFrame
df = pd.DataFrame({
    "Food_ID": np.arange(1, num_rows + 1),
    "Food_Name": food_data,
    "Category": categories,
    "Serving_Size_g": np.random.randint(30, 400, size=num_rows),
    "Calories": np.random.randint(50, 800, size=num_rows),  # Wider calorie range
    "Carbs_g": carbs,
    "Protein_g": protein,
    "Fat_g": fat,
    "Sugar_g": sugar,
    "Fiber_g": fiber,
    "Sodium_mg": np.random.randint(0, 1200, size=num_rows),  # Wider sodium range
    "Cholesterol_mg": cholesterol,
    "Iron_mg": iron,
    "Glycemic_Index": np.random.choice(["Low", "Medium", "High"], size=num_rows, p=[0.4, 0.4, 0.2]),
    "Allergens": np.random.choice(["None", "Nuts", "Dairy", "Gluten", "Shellfish"], 
                                 size=num_rows, p=[0.6, 0.1, 0.15, 0.1, 0.05]),
    "Meal_Type": np.random.choice(["Breakfast", "Lunch", "Dinner", "Snack"], size=num_rows),
    "Price_USD": np.round(np.abs(np.random.normal(3, 2, size=num_rows)), 2)  # Price distribution
})

# Add missing values (5% missing in multiple columns)
missing_cols = ["Glycemic_Index", "Allergens", "Iron_mg"]
for col in missing_cols:
    df[col] = df[col].mask(np.random.random(size=num_rows) < 0.05, np.nan)

# Save to CSV
df.to_csv("nutrition_dataset_3000.csv", index=False)
print("Dataset saved as 'nutrition_dataset_3000.csv'!")