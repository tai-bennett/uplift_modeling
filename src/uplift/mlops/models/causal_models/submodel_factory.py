from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

class SubmodelFactory():
    def create(self, name):
        if name == "sk_gradient_boosting_regressor":
            return GradientBoostingRegressor
        raise ValueError(f"Unknown model of type {name}.")
