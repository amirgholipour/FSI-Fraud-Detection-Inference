import tensorflow as tf
import joblib
import numpy as np
import json
import traceback
import sys
import os



class Predictor(object):

#     def __init__(self):
#         self.model = joblib.load('model.pkl')
    def __init__(self):
        self.loaded = False
        self.model = joblib.load('finalized_ml_model.sav')
        self.class_name = ['None Fraud', 'Fraud']
    def load(self):
        print("Loading model",os.getpid())
        self.model = tf.keras.models.load_model('finalized_dl_model.h5', compile=False)
        self.loaded = True
        print("Loaded model")

        
        
    def predict(self, X,features_names=None):
        # data = request.get("data", {}).get("ndarray")
        # mult_types_array = np.array(data, dtype=object)
#         print ('step1......')
#         print(X)
#         X = tf.constant(X)
#         print ('step2......')
#         print(X)
        if not self.loaded:
            self.load()
#         result = self.model.predict(X)
        try:
                       
            
            print ('Step1:  Perform prediction!!!')
            if not self.loaded:
                pred_prob = self.model.predict(X)
                predicted_class=int(np.round(pred_prob))
            else:
                predicted_class = int(self.model.predict(model_ready_input))
                pred_prob = self.model.predict_proba(model_ready_input)[:, 1]
                # predicted_class=np.round(result)
            print ('Step1 finished!!!!')
            # print(result)
            print(predicted_class)
            pred_label = self.class_nameclass_name[predicted_class]
            print('Predicted Class name: ', pred_label)

    
            # json_results = {"Predicted Class": str(predicted_class)}
            json_results = {"Predicted value": json.dumps(predicted_class.numpy(), cls=JsonSerializer) ,"Predicted Class Label": pred_label,"Predicted Class Probability": pred_prob.tolist()}
        

        
        ######
        except Exception as e:
            print(traceback.format_exception(*sys.exc_info()))
            raise # reraises the exception
                
        
        return json.dumps(result.numpy(), cls=JsonSerializer)    

#     def predict_raw(self, request):
#         data = request.get("data", {}).get("ndarray")
#         mult_types_array = np.array(data, dtype=object)

#         result = self.model.predict(mult_types_array)

#         return json.dumps(result, cls=JsonSerializer)

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


