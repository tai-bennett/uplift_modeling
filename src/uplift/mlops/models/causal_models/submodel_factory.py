from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

class SubmodelFactory():
    def create(self, name):
        if name == "sk_gradient_boosting_regressor":
            return GradientBoostingRegressor
        if name == "sk_random_forest_classifier":
            return RandomForestClassifier
        if name == "xgb_classifier":
            return XGBClassifier
        if name == "xgb_regressor":
            return XGBRegressor
        raise ValueError(f"Unknown model of type {name}.")
