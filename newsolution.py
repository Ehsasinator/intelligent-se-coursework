import pandas as pd
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Systems to evaluate
systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
num_repeats = 3  # Number of repetitions for robustness
train_frac = 0.7  # Fraction of data for training
random_seed = 1  # Random seed for reproducibility

def evaluate_model(train_X, train_Y, test_X, test_Y):
    """Train and evaluate Gradient Boosting, returning performance metrics."""
    model = GradientBoostingRegressor(n_estimators=100, random_state=1)
    model.fit(train_X, train_Y)
    predictions = model.predict(test_X)

    mape = mean_absolute_percentage_error(test_Y, predictions)
    mae = mean_absolute_error(test_Y, predictions)
    rmse = np.sqrt(mean_squared_error(test_Y, predictions))

    return mape, mae, rmse

def main():
    for current_system in systems:
        datasets_location = f'datasets/{current_system}'
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {current_system}, Dataset: {csv_file}, Training Fraction: {train_frac}, Repeats: {num_repeats}")

            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for current_repeat in range(num_repeats):
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                train_X, train_Y = train_data.iloc[:, :-1], train_data.iloc[:, -1]
                test_X, test_Y = test_data.iloc[:, :-1], test_data.iloc[:, -1]

                mape, mae, rmse = evaluate_model(train_X, train_Y, test_X, test_Y)

                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Print average metrics
            print(f"  - Average MAPE: {np.mean(metrics['MAPE']):.4f}")
            print(f"  - Average MAE: {np.mean(metrics['MAE']):.4f}")
            print(f"  - Average RMSE: {np.mean(metrics['RMSE']):.4f}")

if __name__ == "__main__":
    main()
