import numpy as np
import pandas as np

#machine learning libraries
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

obj_bo = 'roc_auc'
cv_splits = 4
gp_params = {'alpha': 1e-4}





def evaluate(y_hat, y):
    #function to evaluate predictions
    print('accuracy: '+str(accuracy_score(y,y_hat)))
    print('roc auc: '+str(roc_auc_score(y,y_hat,average='weighted')))



def treesCV(eta, gamma,max_depth,min_child_weight,subsample,colsample_bytree,n_estimators):
    #function for cross validation gradient boosted trees
    return cross_val_score(xgb.XGBRegressor(objective='binary:logistic',
                                                learning_rate=max(eta,0),
                                                gamma=max(gamma,0),
                                                max_depth=int(max_depth),
                                                min_child_weight=int(min_child_weight),
                                                silent=True,
                                                subsample=max(min(subsample,1),0.0001),
                                                colsample_bytree=max(min(colsample_bytree,1),0.0001),
                                                n_estimators=int(n_estimators),
                                                seed=42), X=X_train, y=y_train1, scoring=obj_bo, cv=cv_splits).mean()


def svmCVl1(C,max_iter=1000,tol=1e-4):
    #function for cross validation svc with l1 penalty
	return cross_val_score(LinearSVC(C=C,tol=tol,penalty='l1',max_iter=max_iter), X=X_train, y=y_train1, scoring=obj_bo, cv=cv_splits).mean()

def svmCVl2(C,max_iter=1000,tol=1e-4):
    #function for cross validation svc with l2 penalty
	return cross_val_score(LinearSVC(C=C,tol=tol,penalty='l2',max_iter=max_iter), X=X_train, y=y_train1, scoring=obj_bo, cv=cv_splits).mean()

def logitCV(C,max_iter=1000,tol=1e-4):
    #function for cross validation svc with l2 penalty
	return cross_val_score(LogisticRegression(C=C,tol=tol,penalty='l2',max_iter=max_iter), X=X_train, y=y_train1, scoring=obj_bo, cv=cv_splits).mean()


