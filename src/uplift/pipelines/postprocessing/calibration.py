import pdb

from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self, metadata, y_min=-0.999, y_max=0.999):
        self.metadata = metadata
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')

    def run(self, model, data):
        self.fit(model, data)
        model.calibration = self.model
        return model


    def fit(self, model, data):
        x = data.select(self.metadata['feature_names']).to_pandas()
        s = model.predict(x)
        y = data.select(self.metadata['target_name']).to_numpy()
        t = data.select(self.metadata['treatment_name']).to_numpy()

        p_t = sum(t) / len(t)
        # confirm this formula using nyberg or other reference
        r = y * (t - p_t) / (p_t * (1 - p_t))
        # r = (y - t) / (p_t * (1 - p_t))
        r = r.flatten()
        pdb.set_trace()
        self.model.fit(s, r)



