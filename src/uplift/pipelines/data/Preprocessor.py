import pdb
from easydict import EasyDict as edict

class Preprocessor():
    def __init__(self, data, params):
        self.data = data
        self.params = edict(params)
        self.schema = {}

    def run(self):
        pdb.set_trace()
        self._infer_schema()
        self._manual_schema()
        self._generate_schema()
        return self.data, self.schema

    def _infer_schema(self):
        pass

    def _manual_schema(self):
        cat_names = self.params.manual_schema.category
        self.data[cat_names] = self.data[cat_names].astype('category')
        cont_names = self.params.manual_schema.continuous
        self.data[cont_names] = self.data[cont_names].astype('float')

    def _generate_schema(self):
        """ from manual schema and dataframe, generate full schema to save"""
        pass

