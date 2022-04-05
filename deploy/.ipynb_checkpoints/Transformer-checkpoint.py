import numpy as np
import joblib
import pandas as pd

class Transformer(object):
    
    def __init__(self):
        self.scaler = joblib.load('scaler.pkl')

    def transform_input(self, X, feature_names =None , meta=None ):
        print(X)
        print('*'*50)
        print('*'*50)
        print('*'*50)
        print(X['data']['ndarray'])

        transformed_data = self.scaler.transform(X['data']['ndarray'])

        # print(df.to_numpy())
        return transformed_data