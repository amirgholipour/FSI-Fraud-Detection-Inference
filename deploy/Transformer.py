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
        print(feature_names)
        print('+'*50)
        print(meta)
        print('-'*50)
        df = pd.DataFrame(X, columns=feature_names)
        print('x'*50)
        print(df)
        print('z'*50)
        
        transformed_data = self.scaler.transform(df)
        
        print(transformed_data)
        print('y'*50)
        # df = pd.DataFrame(X, columns=feature_names)
        # df = self.encoder.transform(df)
        # df = self.onehotencoder.transform(df)
        # # print(df.to_numpy())
        # return df.to_numpy()

        # print(df.to_numpy())
        # return transformed_data.to_numpy()
        return transformed_data