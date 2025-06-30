import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Step 1: Load the CSV
df = pd.read_csv("salaries.csv", encoding='ISO-8859-1')  # ✅ Use your actual file name

# Step 2: Check column names
print("\n✅ Columns:\n", df.columns.tolist())

# Step 3: Define target column — fix name if needed
target_column = 'salary_in_usd'  # ⛳ Change this to exact name if needed
if target_column not in df.columns:
    print(f"\n❌ Column '{target_column}' not found. Check your column names above.")
    exit()

# Step 4: Prepare features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Step 5: One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Define models
models = {
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "Linear Regression": LinearRegression()
}

# Step 8: Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

# Step 9: GridSearchCV for Random Forest
grid_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
grid = GridSearchCV(RandomForestRegressor(), grid_params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
print("\n✅ Best GridSearchCV Params (Random Forest):", grid.best_params_)

# Step 10: RandomizedSearchCV for SVR
rand_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'epsilon': [0.1, 0.2]
}
random = RandomizedSearchCV(SVR(), rand_params, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42)
random.fit(X_train, y_train)
print("✅ Best RandomizedSearchCV Params (SVR):", random.best_params_)
