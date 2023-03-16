# some general code that we will use across the entire project(like reading the dataset)
import os,sys

import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params,cv=3,n_jobs=3,verbose=1,refit=False):
    try:
        report={}
        for i in range(len(models)):
            
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
            gs.fit(X_train,y_train)
            
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]]=test_model_score
     

        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
    