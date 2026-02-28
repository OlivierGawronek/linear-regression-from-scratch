import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    return x, y

def cost_calculate(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = (1 / (2 * m)) * cost
    return total_cost

def gradient_calculate(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, learning_rate, num_iters):
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_calculate(x, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        cost_history.append(cost_calculate(x, y, w, b))
    return w, b, cost_history

def save_cost_history(cost_history, filepath="out/cost_history.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "cost"])
        for i, c in enumerate(cost_history):
            writer.writerow([i, c])

def main():
    filepath = input("Path to CSV file (default: source/sample.csv): ") or "source/sample.csv"
    x_train, y_train = load_data(filepath)
    print(f"Loaded {len(x_train)} data points.")

    w = float(input("Initial w (default 0): ") or "0")
    b = float(input("Initial b (default 0): ") or "0")
    alpha = float(input("Learning rate (default 0.01): ") or "0.01")
    num_iters = int(input("Number of iterations (default 1000): ") or "1000")

    w_final, b_final, cost_history = gradient_descent(
        x_train, y_train, w, b, alpha, num_iters
    )

    print(f"\nResults:")
    print(f"  w: {w_final:.4f}")
    print(f"  b: {b_final:.4f}")
    print(f"  Final cost: {cost_history[-1]:.4f}")

    save_cost_history(cost_history)
    print("Cost history saved to out/cost_history.csv")

if __name__ == "__main__":
    main()