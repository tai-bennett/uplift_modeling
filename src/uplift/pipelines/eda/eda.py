import pandas as pd
import plotly.express as px
from uplift.config.loaders import get_paths
from .utils import get_numeric_columns, get_category_columns, save_figure
#from umap.umap_ import umap
import umap
import pdb
import pprint
from easydict import EasyDict as edict

class EDA():
    def __init__(self, ds, schema, params):
        self.df = ds['train'].to_pandas()
        self.schema = edict(schema)
        self.params = edict(params)
        self.results = {}
        self.save_root = get_paths()['eda_report']
        #self.numeric_cols = get_numeric_columns(self.df)
        #self.cat_cols = get_category_columns(self.df)

    def run(self):
        self._preprocess()
        self._missingness()
        self._feature_plots()
        self._class_analysis()
        self._cor_analysis()
        self._outlier_analysis()
        #self._umap()

    def _preprocess(self):
        self.df['treat_conv'] = pd.Categorical(
            list(zip(self.df['treatment'], self.df['conversion']))
        )
        for name in ['treatment', 'conversion', 'exposure']:
            self.df[name] = self.df[name].astype('category')

        self.feat_names = [f'f{i}' for i in range(12)]
        
    def _missingness(self):
        results = {}
        self.results['missingness'] = results

    def _feature_plots(self):
        for name in self.schema.target_names:
            self._histogram(name)

    def _histogram(self, target):
        for name in self.schema.feature_names:
            fig = px.histogram(
                self.df,
                x=name,
                color=target,
                marginal='box',
                barmode='overlay',
                nbins=50,
                opacity=0.7,
                title=f"diistribution of {name}"
                )

            save_figure(fig, self.save_root, f"histogram_{target}_vs_{name}")

            
            

    def _class_analysis(self):
        results = {}
        treat_counts = self.df['treatment'].value_counts().reset_index()
        treat_counts.columns = ['treatment', 'counts']
        results['treat_counts'] = treat_counts.values.tolist()

        conversion_counts = self.df['conversion'].value_counts().reset_index()
        conversion_counts.columns = ['conversion', 'counts']
        results['conversion_counts'] = conversion_counts.values.tolist()

        treat_conv_counts = self.df['treat_conv'].value_counts().reset_index()
        treat_conv_counts.columns = ['treat_conv', 'counts']
        results['treat_conv_counts'] = conversion_counts.values.tolist()
        
        fig_treatment = px.bar(
            treat_counts,
            x='treatment',
            y='counts',
            title='Treatment Counts'
            )

        save_figure(fig_treatment, self.save_root, "conversion_counts")

        fig_conversion = px.bar(
            conversion_counts,
            x='conversion',
            y='counts',
            title='Conversion Counts'
            )

        save_figure(fig_conversion, self.save_root, "conversion_counts")

        fig_treat_conv = px.bar(
            treat_conv_counts,
            x='treat_conv',
            y='counts',
            title='(Treatment, Conversion) Counts'
            )

        save_figure(fig_treat_conv, self.save_root, "treatment_counts")

        # pdb.set_trace()

        # fig_combo = px.bar(
        #     combo_counts,
        #     x='class',
        #     y='counts',
        #     title='Class Counts'
        #     )

        # save_figure(fig_treat, self.save_root, "treatment_counts")

        self.results['class'] = results

    def _cor_analysis(self):
        results = {}
        cor = self.df[self.feat_names].corr()

        fig = px.imshow(
            cor,
            text_auto=False,
            aspect='auto',
            title="Correlation Heatmap"
            )

        correlated = {}
        for i in range(len(self.schema.feature_names)):
            for j in range(i):
                if abs(cor.iloc[i, j]) > self.params.correlation.threshold:
                    correlated[(i, j)] = cor.iloc[i, j]

        results['high_correlation'] = correlated
        save_figure(fig, self.save_root, "correlation_heatmap")
        self.results['cor'] = results

    def _outlier_analysis(self):
        results = {}
        self.results['outlier'] = results

    def _umap(self):
        results = {}
        reducer = umap.UMAP(**self.params.umap.reducer)
        X = reducer.fit_transform(self.df[self.feat_names])
        umap_df = pd.DataFrame({
            'umap_x': X[:, 0],
            'umap_y': X[:, 1],
            'treatment': self.df.treatment,
            'treat_conv': self.df.treat_conv,
            'conversion': self.df.conversion
            })

        fig = px.scatter(
            umap_df,
            x='umap_x',
            y='umap_y',
            color='treatment',
            opacity=self.params.umap.plot.opacity
            )

        save_figure(fig, self.save_root, "umap_treatment")

        fig = px.scatter(
            umap_df,
            x='umap_x',
            y='umap_y',
            color='treat_conv',
            opacity=self.params.umap.plot.opacity
            )

        save_figure(fig, self.save_root, "umap_treat_conv")
        self.results['umap'] = results
        


def main(ds, schema, params):
    eda = EDA(ds, schema, params)
    eda.run()
    pprint.pp(eda.results)
    return eda.results


