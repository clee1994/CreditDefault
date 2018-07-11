import numpy as np
import pandas as pd


#machine learning libraries
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

iter_no = 10
gp_params = {'alpha': 1e-4}
obj_bo = 'roc_auc'
cv_splits = 4


def treesCV(eta, gamma,max_depth,min_child_weight,subsample,colsample_bytree,n_estimators):
    #function for cross validation gradient boosted trees
    return cross_val_score(xgb.XGBRegressor(objective='binary:logistic',
    											tree_method = 'hist',
                                                learning_rate=max(eta,0),
                                                gamma=max(gamma,0),
                                                max_depth=int(max_depth),
                                                min_child_weight=int(min_child_weight),
                                                silent=True,
                                                subsample=max(min(subsample,1),0.0001),
                                                colsample_bytree=max(min(colsample_bytree,1),0.0001),
                                                n_estimators=int(n_estimators),
                                                seed=42), X=X_train, y=y_train, scoring=obj_bo, cv=cv_splits).mean()


def svmCVl1(C,max_iter=1000,tol=1e-4):
    #function for cross validation svc with l1 penalty
	return cross_val_score(LinearSVC(C=C,tol=tol,penalty='l1',max_iter=max_iter), X=X_train, y=y_train, scoring=obj_bo, cv=cv_splits).mean()

def svmCVl2(C,max_iter=1000,tol=1e-4):
    #function for cross validation svc with l2 penalty
	return cross_val_score(LinearSVC(C=C,tol=tol,penalty='l2',max_iter=max_iter), X=X_train, y=y_train, scoring=obj_bo, cv=cv_splits).mean()

def logitCV(C,max_iter=1000,tol=1e-4):
    #function for cross validation svc with l2 penalty
	return cross_val_score(LogisticRegression(C=C,tol=tol,penalty='l2',max_iter=max_iter), X=X_train, y=y_train, scoring=obj_bo, cv=cv_splits).mean()



def data_prep(data_df):

    #how to handle types
    data_df_num = data_df.select_dtypes(exclude=object)
    data_df_obj = data_df.select_dtypes(include=object)

    #how to handle nan
    data_df_num = data_df_num.fillna(data_df_num.mean())
    #data_df_obj = data_df_obj.fillna(data_df_obj.mode())

    #get dummy variables -> maybe reduce dimensionality here
    data_df_obj = pd.get_dummies(data_df_obj, dummy_na=True)


    return pd.concat([data_df_num, data_df_obj],axis=1)


# reading data
data_train = pd.read_csv('data/application_train.csv', sep=',')
data_train = data_prep(data_train)

data_pred = pd.read_csv('data/application_test.csv', sep=',')
data_pred = data_prep(data_pred)

data_train = data_train.drop(list(set(list(data_train))-set(list(data_pred))-set(['TARGET'])),axis=1)


#train test split doesnt actually split
X_train, X_test, y_train, y_test = train_test_split(np.array(data_train.drop(['TARGET','SK_ID_CURR'],axis=1)), np.array(data_train['TARGET']), test_size=0, random_state=42)
X_test = data_pred.drop(['SK_ID_CURR'],axis=1)



# feauture creation




# models

#Bayesian Hyper parameter optimization of logistic regression l2 penalty
# logitBO = BayesianOptimization(logitCV,{'C':(0.1,5)})
# logitBO.maximize(n_iter=iter_no, **gp_params)
# logitBO_best = logitBO.res['max']
# logit_best = LogisticRegression(**logitBO_best['max_params'],penalty='l2')
# logit_best.fit(X_train, y_train)
# y_hat = logit_best.predict_proba(X_test)[:,1]


# #Bayesian Hyper parameter optimization of support vector machine - l1 penalty
# svcl1BO = BayesianOptimization(svmCVl1,{'C':(0.1,5),'max_iter':(100,10000),'tol':(1e-6,1e-1)})
# svcl1BO.maximize(n_iter=iter_no, **gp_params)
# svcl1BO_best = svcl1BO.res['max']
# svcl1_best = LinearSVC(**svcl1BO_best['max_params'],penalty='l1')
# svcl1_best.fit(X_train, y_train)
# y_hat = svcl1_best.predict(X_test)


# #Bayesian Hyper parameter optimization of support vector machine - l2 penalty
# svcl2BO = BayesianOptimization(svmCVl2,{'C':(0.1,5),'max_iter':(100,10000),'tol':(1e-6,1e-1)})
# svcl2BO.maximize(n_iter=iter_no, **gp_params)
# svcl2BO_best = svcl2BO.res['max']
# svcl2_best = LinearSVC(**svcl2BO_best['max_params'],penalty='l2')
# svcl2_best.fit(X_train, y_train)
# y_hat = svcl2_best.predict(X_test)


#Bayesian Hyper parameter optimization of gradient boosted trees
treesBO = BayesianOptimization(treesCV,{'eta':(0.001,1),
                                        'gamma':(0.0001,10),
                                        'max_depth':(0,80),
                                        'min_child_weight':(0.01,10),
                                        'subsample':(0,1),
                                        'colsample_bytree':(0,1),
                                        'n_estimators':(10,300)})
treesBO.maximize(n_iter=iter_no, **gp_params)
tree_best = treesBO.res['max']
trees_model = xgb.XGBRegressor(objective='binary:logistic',
								tree_method = 'hist',
                                seed=42,
                                learning_rate=max(tree_best['max_params']['eta'],0),
                                gamma=max(tree_best['max_params']['gamma'],0),
                                max_depth=int(tree_best['max_params']['max_depth']),
                                min_child_weight=int(tree_best['max_params']['min_child_weight']),
                                silent=True,
                                subsample=max(min(tree_best['max_params']['subsample'],1),0.0001),
                                colsample_bytree=max(min(tree_best['max_params']['colsample_bytree'],1),0.0001),
                                n_estimators=int(tree_best['max_params']['n_estimators']))
trees_model.fit(X_train, y_train)
y_hat = trees_model.predict(np.array(X_test))








#write to file for submission
submission = pd.DataFrame({'SK_ID_CURR':np.array(data_pred['SK_ID_CURR']),'TARGET':y_hat})
submission.to_csv('submission.csv',sep=',', index=False)


