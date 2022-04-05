import tensorflow as tf
import joblib
import numpy as np
import json
import traceback
import sys
import os



class Predictor(object):

    def __init__(self,loaded = False):
        self.loaded = loaded
        if self.loaded == True:
            self.model = joblib.load('finalized_ml_model.sav')
        self.class_names = ['None Fraud', 'Fraud']
    def load(self):
        print("Loading model",os.getpid())
        self.model = tf.keras.models.load_model('finalized_dl_model.h5', compile=False)
        self.loaded = True
        print("Loaded model")

        
        
    def predict(self, X,features_names=None):

        try:
            print ('Step1:  Perform prediction!!!')
            if not self.loaded:
                print ('Loading DL model!!!!')
                self.load()
                pred_prob = self.model.predict(X)
                predicted_class=int(np.round(pred_prob))
            else:
                print("Loading ML model!!!!  ")
                predicted_class = int(self.model.predict(X))
                
                pred_prob = self.model.predict_proba(X)#[:, 1]
            print ('Step1 finished!!!!')

        
            print(predicted_class)
            pred_label = self.class_names[predicted_class]
            print('Predicted Class name: ', pred_label)

            json_results = {"Predicted value": json.dumps(predicted_class, cls=JsonSerializer) ,"Predicted Class Label": pred_label,"Predicted Class Probability": pred_prob.tolist()}

        
        ######
        except Exception as e:
            print(traceback.format_exception(*sys.exc_info()))
            raise # reraises the exception
                
        
        return json.dumps(json_results)    



class JsonSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
        np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)