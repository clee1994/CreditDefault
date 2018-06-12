import numpy as np
import pandas as pd


#machine learning libraries
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC


from cv_estimators import treesCV, svmCVl1, svmCVl2, logitCV, evaluate





iter_no = 25


# reading data
data_train = pd.read_csv('data/application_train.csv', sep=',')


#fill strategy
data_train = data_train.dropna()


X_train, X_test, y_train, y_test = train_test_split(np.array(data_train.drop(['TARGET','SK_ID_CURR'],axis=1)), np.array(data_train['TARGET']), test_size=0.33, random_state=42)

# feauture creation




# models

#Bayesian Hyper parameter optimization of support vector machine - l1 penalty
svcl1BO = BayesianOptimization(logitCVl1,{'C':(0.1,5)})
svcl1BO.maximize(n_iter=iter_no, **gp_params)
svcl1BO_best = svcl1BO.res['max']
svcl1_best = LinearSVC(**svcl1BO_best['max_params'],penalty='l1')
svcl1_best.fit(X_train, y_train1)
y_hat_svcl1 = svcl1_best.predict(X_test)




#Bayesian Hyper parameter optimization of support vector machine - l1 penalty
svcl1BO = BayesianOptimization(svmCVl1,{'C':(0.1,5),'max_iter':(100,10000),'tol':(1e-6,1e-1)})
svcl1BO.maximize(n_iter=iter_no, **gp_params)
svcl1BO_best = svcl1BO.res['max']
svcl1_best = LinearSVC(**svcl1BO_best['max_params'],penalty='l1')
svcl1_best.fit(X_train, y_train1)
y_hat_svcl1 = svcl1_best.predict(X_test)



#Bayesian Hyper parameter optimization of support vector machine - l2 penalty
svcl2BO = BayesianOptimization(svmCVl2,{'C':(0.1,5),'max_iter':(100,10000),'tol':(1e-6,1e-1)})
svcl2BO.maximize(n_iter=iter_no, **gp_params)
svcl2BO_best = svcl2BO.res['max']
svcl2_best = LinearSVC(**svcl2BO_best['max_params'],penalty='l2')
svcl2_best.fit(X_train, y_train1)
y_hat_svcl2 = svcl2_best.predict(X_test)



#Bayesian Hyper parameter optimization of gradient boosted trees
treesBO = BayesianOptimization(treesCV,{'eta':(0.0001,1),
                                        'gamma':(0.0001,100),
                                        'max_depth':(0,300),
                                        'min_child_weight':(0.001,10),
                                        'subsample':(0,1),
                                        'colsample_bytree':(0,1),
                                        'n_estimators':(10,1000)})
treesBO.maximize(n_iter=iter_no, **gp_params)
tree_best = treesBO.res['max']
trees_model = xgb.XGBRegressor(objective='multi:softmax',
                                seed=42,
                                learning_rate=max(tree_best['max_params']['eta'],0),
                                gamma=max(tree_best['max_params']['gamma'],0),
                                max_depth=int(tree_best['max_params']['max_depth']),
                                min_child_weight=int(tree_best['max_params']['min_child_weight']),
                                silent=True,
                                subsample=max(min(tree_best['max_params']['subsample'],1),0.0001),
                                colsample_bytree=max(min(tree_best['max_params']['colsample_bytree'],1),0.0001),
                                n_estimators=int(tree_best['max_params']['n_estimators']))
trees_model.fit(X_train, y_train1)
y_hat_trees = trees_model.predict(X_test)




# predicting
data_pred = pd.read_csv('application_test.csv', sep=',')











#print results
print('------ Results: SVC l1 ------')
evaluate(y_hat_svcl1,y_test1)
print('------ Results: SVC l2 ------')
evaluate(y_hat_svcl2,y_test1)
print('------ Results: XGB ------')
evaluate(y_hat_trees,y_test1)
