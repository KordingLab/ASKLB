"""
Unit tests for asklb.automl module.
"""

import pandas as pd
import numpy as np
import unittest

import asklb.automl

class TestAutoML(unittest.TestCase):

    def test_transform_features(self):
        n = 100
        numeric_col = np.random.choice(2, size=n)
        df = pd.DataFrame()
        
        df['numeric'] = numeric_col
        df['str_cat'] = numeric_col
        df['str_cat'] = df['str_cat'].map({0: "female", 1: "male"})
        #df['obj_cat'] = map(lambda x: [] if x > 0 else {}, numeric_col)

        data, feat_types, cat_dict = asklb.automl.process_feat_types(df)

        self.assertEqual(data['str_cat'].dtype, np.int64)
        self.assertEqual(feat_types, ['Numerical', 'Categorical'])
        self.assertEqual(sorted(cat_dict['str_cat']), ["female", "male"])
