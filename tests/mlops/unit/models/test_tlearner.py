import numpy as np
import polars as pl

# Main imports
# Helper imports
from sklearn.model_selection import train_test_split

from uplift.mlops.models.causal_models.tlearner import *
from uplift.mlops.training_data import PolarsData


# Define DGP
def test_tlearner():
    # generate synthetic data
    data, data_test = generate_data()

    # set up models
    params = {
        "n_estimators": 2,
        "max_depth": 2,
        "learning_rate": 1,
        "objective": "binary:logistic",
    }
    model = MyTLearner("xgb_classifier", params)
    model.fit(data)
    uplift_test = model.eval(data_test["features"])


def generate_data():
    rng = np.random.default_rng(42069)
    n = 10000
    p = 5
    hold_out_ratio = 0.2
    train_size = n * (1 - p)

    # random covariates
    cluster = rng.choice([0, 1, 2], size=n, p=[0.4, 0.3, 0.3])

    means = np.array([[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [-2, -2, -2, -2, -2]])

    cov = np.eye(p)

    X = np.zeros((n, p))

    for k in range(3):
        idx = cluster == k
        X[idx] = rng.multivariate_normal(means[k], cov, size=idx.sum())

    # random treatment
    T = rng.binomial(1, 0.5, size=n)

    logit_control = -1.5 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]

    p_control = 1 / (1 + np.exp(-logit_control))

    true_uplift = np.select(
        [cluster == 0, cluster == 1, cluster == 2], [0.15, -0.05, 0.30]
    )

    p_treatment = np.clip(p_control + true_uplift, 0.001, 0.999)

    p_observed = np.where(T == 1, p_treatment, p_control)

    Y = rng.binomial(1, p_observed)

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=hold_out_ratio, random_state=42069
    )

    test_data = {"features": X_test, "treatment": T_test, "target": Y_test}
    names = []
    metadata = {}
    for i in range(p):
        names.append(f"feature_{i}")
    metadata["feature_names"] = names.copy()
    names.append("treatment")
    names.append("conversion")
    metadata["treatment_name"] = "treatment"
    metadata["target_name"] = "conversion"

    df = np.append(X_train, T_train.reshape((train_size, 1)), axis=1)
    df = np.append(df, Y_train.reshape((train_size, 1)), axis=1)
    df = pl.DataFrame(df, schema=names)

    df = PolarsData(df, metadata)
    return df, test_data
