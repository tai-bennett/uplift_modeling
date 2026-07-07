from uplift.mlops.models import ModelFactory


def test_model_factory():
    causal_model_name = "tlearner"
    model_name = "sk_gradient_boosting_regressor"
    params = {"n_estimators": 12, "max_depth": 6, "min_samples_leaf": 5}

    _ = ModelFactory().create(causal_model_name)(model_name, params)
