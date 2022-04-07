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
        #print(X['data']['ndarray'])
        print(feature_names)
        print(X.to_numpy())
        
        X = pd.DataFrame(X, columns=feature_names)
        transformed_data = self.scaler.transform(X)
        


        
        #transformed_data = self.scaler.transform(X['data']['ndarray'])
        
#         transformed_data = self.scaler.transform(X)
        print('finished transformation')
        # print(df.to_numpy())
        return transformed_data.to_numpy()