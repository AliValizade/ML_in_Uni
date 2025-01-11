import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Read dataset
def read_dataset():
    X = pd.read_csv('data/Xtrain.csv').values.astype(np.float32)
    y = pd.read_csv('data/Ytrain.csv').values.astype(np.float32)
    return X, y

# Ridge Regression using formula
def ridge_regression(X, y, lambda_val=0.1):
    X_transpose = X.T
    identity_matrix = np.eye(X.shape[1])
    w = np.linalg.inv(X_transpose.dot(X) + lambda_val * identity_matrix).dot(X_transpose).dot(y)
    return w

# Custom function to create polynomial features as specified
def create_custom_polynomial_features(X):
    X_poly = np.ones((X.shape[0], 1))  # Start with x^0
    for i in range(1, 4):  # x^1 to x^3
        X_poly = np.hstack((X_poly, X ** i))
    return X_poly

# Function to compute error metrics
def compute_errors(w, X, y):
    y_pred = X.dot(w)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return mse, mae

def run_experiment():
    # Read dataset
    X, y = read_dataset()

    # Create custom polynomial features
    X_poly = create_custom_polynomial_features(X)

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_lambda = 0.1
    best_mse = float('inf')
    best_mae = float('inf')
    lambdas = [0.01, 0.1, 1, 10]

    # Search for best lambda using cross-validation
    for lambda_val in lambdas:
        mse_scores = []
        mae_scores = []
        for train_index, test_index in kf.split(X_poly):
            # Split data
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # # Normalize data
            # scaler = StandardScaler()
            # X_train_normalized = scaler.fit_transform(X_train)
            # X_test_normalized = scaler.transform(X_test)

            # Train model
            w = ridge_regression(X_train, y_train, lambda_val)

            # Compute errors
            mse, mae = compute_errors(w, X_test, y_test)
            mse_scores.append(mse)
            mae_scores.append(mae)

        # Average MSE and MAE for current lambda
        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)

        # Update best lambda if lower MSE is found
        if avg_mse < best_mse:
            best_lambda = lambda_val
            best_mse = avg_mse
            best_mae = avg_mae

    # Final model training with best lambda on full dataset
    w_final = ridge_regression(X_poly, y, best_lambda)

    print(f"Best lambda: {best_lambda}")
    print(f"Best Cross-Validated MSE: {best_mse:.4f}")
    print(f"Best Cross-Validated MAE: {best_mae:.4f}")

run_experiment()

