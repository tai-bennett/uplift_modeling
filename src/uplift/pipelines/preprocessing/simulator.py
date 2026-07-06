import numpy as np
import polars as pl


class Simulator:
    def __init__(self, revenue_mu=4, revenue_sigma=0.75):
        self.mu = revenue_mu
        self.sigma = revenue_sigma

    def run(self, ds, meta):
        revenue = np.random.lognormal(mean=self.mu, sigma=self.sigma, size=ds.height)
        cost = np.random.gamma(2, scale=1, size=ds.height)

        ds = ds.with_columns(
            pl.when(pl.col(meta['target_name']) == 1).then(pl.Series(revenue)).otherwise(0.0).alias("revenue")
            )
        ds = ds.with_columns(
            pl.when(pl.col(meta['treatment_name']) == 1).then(pl.Series(cost)).otherwise(0.0).alias("cost")
            )
        meta['revenue_name'] = 'revenue'
        meta['cost_name'] = 'cost'
        return ds, meta
