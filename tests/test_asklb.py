"""
Unit tests for asklb.automl module.
"""

import pandas as pd
import numpy as np
import unittest

import asklb.model_utils

class TestModelUtils(unittest.TestCase):

    def test_transform_features(self):
        numeric_col = np.random.choice(2, size=100)
        df = pd.DataFrame()
        
        df['numeric'] = numeric_col
        df['str_cat'] = numeric_col
        df['str_cat'] = df['str_cat'].map({0: "female", 1: "male"})

        data, feat_types, cat_dict = asklb.model_utils.process_feat_types(df)

        self.assertEqual(data['str_cat'].dtype, np.int64)
        self.assertEqual(feat_types, ['Numerical', 'Categorical'])
        self.assertEqual(sorted(cat_dict['str_cat']), ["female", "male"])
