# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lingam import DirectLiNGAM
from sklearn.linear_model import LinearRegression

# 2. Data generating process
def generate_data(n, non_gaussian=False):
    if non_gaussian:
        X = np.random.laplace(size=n)
        epsilon = np.random.laplace(size=n)
    else:
        X = np.random.normal(size=n)
        epsilon = np.random.normal(size=n)
    
    Y = 1.0 * X + epsilon
    return X.reshape(-1, 1), Y.reshape(-1, 1)

# 3. LiNGAM estimation
def lingam_direction(X, Y):
    model = DirectLiNGAM()
    data = np.hstack([X, Y])
    model.fit(data)
    return model.adjacency_matrix_

# 4. Mean independence heuristic
def mean_independence_direction(X, Y):
    reg_xy = LinearRegression().fit(X, Y)
    resid_xy = Y - reg_xy.predict(X)

    reg_yx = LinearRegression().fit(Y, X)
    resid_yx = X - reg_yx.predict(Y)

    corr_xy = np.corrcoef(X.flatten(), resid_xy.flatten())[0,1]
    corr_yx = np.corrcoef(Y.flatten(), resid_yx.flatten())[0,1]

    return abs(corr_xy) < abs(corr_yx)

# 5. Simulation loop
def run_simulation(n, reps, non_gaussian):
    lingam_success = []
    meanind_success = []

    for _ in range(reps):
        X, Y = generate_data(n, non_gaussian)

        # True: X -> Y
        adj = lingam_direction(X, Y)
        lingam_success.append(adj[0,1] != 0)

        meanind_success.append(mean_independence_direction(X, Y))

    return np.mean(lingam_success), np.mean(meanind_success)

# 6. Main
if __name__ == "__main__":
    results = []

    for dist in ["gaussian", "non_gaussian"]:
        for n in [100, 300, 1000]:
            lingam_rate, meanind_rate = run_simulation(
                n=n,
                reps=100,
                non_gaussian=(dist == "non_gaussian")
            )

            results.append({
                "scenario": dist,
                "n": n,
                "lingam_success_rate": lingam_rate,
                "meanind_success_rate": meanind_rate
            })

    df = pd.DataFrame(results)
    print(df)

    # Save table
    df.to_csv("outputs/tables/results.csv", index=False)

    # Plot
    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario]

        plt.plot(subset["n"], subset["lingam_success_rate"], label="LiNGAM")
        plt.plot(subset["n"], subset["meanind_success_rate"], label="MeanInd")

        plt.title(f"Success Rate - {scenario}")
        plt.xlabel("Sample size")
        plt.ylabel("Success rate")
        plt.legend()
        plt.savefig(f"outputs/figures/{scenario}.png")
        plt.clf()
