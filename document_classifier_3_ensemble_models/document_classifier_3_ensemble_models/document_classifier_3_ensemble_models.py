
# document_classifier_3_ensemble_models
# The augmented dataset now has the follow classes distribution (0 = 3984, 1 = 1935, 2 = 550).
# The goal of this script is to implement and test a random forest and a xgboost model for the classification of texts.

import pandas as pd 
import numpy as np 
import xgboost as xgb 
import time, json, pikle   
from collections import Counter 
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder  
from sklearn.experimental import enable_halving_search_cv    
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV, RepeatedStratifiedKFold 
from sklearn.decomposition import PCA, TruncatedSVD 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.metrics import accuracy_score, log_loss, f1_score, balanced_accuracy_score  


def splitData ( df ):
    kl = np.unique( np.array(df['Labels']))
    df_training = pd.DataFrame()
    df_test = pd.DataFrame()
    for k in kl:
        df_class = df[ df['Labels'] == k]
        n = int( df_class.shape[0]/10)
        df_tr = df_class[:-n]
        df_ts = df_class[-n:]
        df_training= pd.concat([df_training, df_tr], axis=0)
        df_test = pd.concat([ df_test, df_ts], axis=0)
    for i in range(0, 3):
        df_training= shuffle( df_training, random_state=33)
        df_test = shuffle(df_test, random_state=33)
    return df_training, df_test

print("go ")
df = pd.read_excel('...\\df_ramo_corpus_big_with_keywords.xlsx')
print("df size ",df.shape)
print(" true labels are: ",Counter(df['Ramo']))

# Transform categorical labels into  numbers
df = df[['corpus', 'Ramo']]
le = LabelEncoder()
labels  = le.fit_transform(df['Ramo'])
df['Labels'] = labels 
df = shuffle( df, random_state=52 )

# Split the dataset into training set and test set and mix the rows 
df_training, df_test = splitData( df )
print(" training ",df_training.shape," test ",df_test.shape)
print(" true labels in test set ",Counter(df_test['Labels'])) 

# Create the Tf-Idf matrix and apply  the latent semantic analysis  to reduce the features 
tv = TfidfVectorizer(min_df=10, norm='l2', use_idf=True )
x_set = tv.fit_transform( df_training['corpus'])
x_set = x_set.toarray()
voc = tv.get_feature_names()
print(" the nuber of words are ",len(voc))
svd = TruncatedSVD( n_components=2000, random_state=42)
x_set = svd.fit_transform(x_set )
print(" svd reductions",x_set.shape)
y_set = df_training['Labels'].values 


# Apply GridSearch to training random forest  to find the model with the best parameters
rfc = RandomForestClassifier( class_weight='balanced_subsample')
rfc_params = {
    "n_estimators":[  30, 50], 
    "max_depth": [ 120, 100,  80],
    "max_features":[ 0.8, 0.3, 0.5  ]    
    }

rcv = StratifiedKFold( n_splits=5, shuffle=True, random_state= 42 )
hgs = HalvingGridSearchCV( rfc, rfc_params ,cv=rcv, min_resources='exhaust',  factor=3, aggressive_elimination=False, scoring="f1_weighted", n_jobs=-1, verbose=1) 
hgs.fit(x_set, y_set)
best_values_rfc = hrs.best_params_
print(" the best params are: ",best_values_rfc)
# test the best model in training set  and in the test set  
rfc = hrs.best_estimator_
y_pred = rfc.predict( x_set)
ac_tr = accuracy_score(y_set, y_pred)
f1_tr = f1_score( y_set, y_pred, average='weighted')
bas_tr = balanced_accuracy_score( y_set, y_pred)
y_prob = rfc.predict_proba( x_set)
lg_tr = log_loss( y_set, y_prob)
print(" random forest training: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_tr, lg_tr, f1_tr, bas_tr))

x_ts = tv.transform( df_test['corpus'])
x_ts = x_ts.toarray()
x_ts = svd.transform( x_ts)
y_ts = df_test['Labels'].values
print(" x_ts ",x_ts.shape," y_ts ",y_ts.shape)
y_pred = rfc.predict( x_ts)
ac_rfc = accuracy_score(y_ts, y_pred)
f1_rfc = f1_score( y_ts, y_pred, average='weighted')
bas_rfc = balanced_accuracy_score( y_ts, y_pred)
y_prob = rfc.predict_proba( x_ts)
lg_rfc = log_loss( y_ts, y_prob)
print(" predicted labels",Counter(y_pred))
print("random forest test : accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_rfc, lg_rfc, f1_rfc, bas_rfc))

# Apply the CalibratedClassifierCV   on the model to improve the accuracy on the unbalanced data
cl_rfc = CalibratedClassifierCV(xgb_cl, method='isotonic', cv=5)
cl_svm.fit( x_set, y_set)
y_pr = cl_rfc.predict( x_ts)
ac_cl = accuracy_score( y_ts, y_pr)
f1_cl = f1_score( y_ts, y_pr, average='weighted')
bas_cl = balanced_accuracy_score  (y_ts, y_pr)
y_pb = cl_rfc.predict_proba( x_ts)
lg_cl = log_loss( y_ts, y_pb)
print(" CalibratedClassifier test: accuracy%.4f, loss %.4f, f1 %.4f, bca %.4f " %(ac_cl, lg_cl, f1_cl, bas_cl))


# xgboost version 1 
# Training xgboost classifier with cross validation 
sw = compute_sample_weight( class_weight='balanced', y= y_set) 
xb = xgb.DMatrix( data=x_set, label=y_set)
xgb_cl1 = xgb.XGBClassifier( max_depth=10, colsample_bylevel= 0.5, learning_rate=0.05, objective='multi:softmax', num_class=3, use_label_encoder=False, seed=33)
myParams = xgb_cl.get_xgb_params()

df_xgb= xgb.cv(dtrain=xb,params=myParams,num_boost_round=240,  nfold=5, stratified=True, metrics='mlogloss', seed=33, as_pandas=True, verbose_eval=1)    
print(" df_xgb size  ",df_xgb.shape)
print(df_xgb.head())
# find the numbers of estimators with the best performances and training the model  again  
nc = list(df_xgb)[2]
lc =list( df_xgb[nc]) 
val_min = lc.index( min(lc)) 
print( the best accuracy is with ",val_min," estimators ") "
xgb_cl1.set_params(n_estimators= val_min) 
xgb_cl1.fit(x_set, y_set,sample_weight=sw )

# test the model 
y_pred = xgb_cl1.predict( x_ts)
ac_xgb1 = accuracy_score(y_ts, y_pred)
f1_xgb1 = f1_score( y_ts, y_pred, average='weighted')
bas_xgb1 = balanced_accuracy_score( y_ts, y_pred)
y_prob = xgb_cl.predict_proba( x_ts)
lg_xgb1 = log_loss( y_ts, y_prob)
print(" predict labels ",Counter(y_pred))
print("xgboost 1 test: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_xgb1, lg_xgb1, f1_xgb1, bas_xgb1))


# xgboost version 2 
# Apply GridSearch to training xgboost classifier to find the model with the best parameters
xgb_cl2 = xgb.XGBClassifier( objective='multi:softmax', num_class=3, use_label_encoder=False, seed=33)
xgbParams = {'max_depth':[10, 5 ],
    'colsample_bylevel':[0.5,0.75, 0.3], 
    'learning_rate':[0.05],
    'n_estimators': [130, 180, 150 ],
    'subsample':[0.8, 1]}

rcv = StratifiedKFold( n_splits=5, shuffle=True, random_state= 42 )
hgs_xgb = HalvingGridSearchCV( xgb_cl2, xgbParams,cv=rcv, min_resources='exhaust',  factor=3, aggressive_elimination=False, scoring="f1_weighted", n_jobs=-1, verbose=1) 
hgs_xgb.fit(x_set, y_set,sample_weight=sw )
best_values_xgb = hgs_xgb.best_params_
print(" the best params are: ",best_values_xgb)

# get the best model and evaluate 
xgb_cl2 = hgs_xgb.best_estimator_
y_pred2 = xgb_cl2.predict( x_set)
ac_xgb_tr2 = accuracy_score(y_set, y_pred2)
f1_xgb_tr2 = f1_score( y_set, y_pred2, average='weighted')
bas_xgb_tr2 = balanced_accuracy_score( y_set, y_pred2)
y_prob2 = xgb_cl2.predict_proba( x_set)
lg_xgb_tr2 = log_loss( y_set, y_prob2)
print("xgboost version 2 training: accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_xgb_tr2, lg_xgb_tr2, f1_xgb_tr2, bas_xgb_tr2))

y_pred = xgb_cl2.predict( x_ts)
ac_xgb_ts2 = accuracy_score(y_ts, y_pred)
f1_xgb_ts2 = f1_score( y_ts, y_pred, average='weighted')
bas_xgb_ts2 = balanced_accuracy_score( y_ts, y_pred)
y_prob = xgb_cl2.predict_proba( x_ts)
lg_xgb_ts2 = log_loss( y_ts, y_prob)
print(" predicted labels ",Counter(y_pred))
print(" xgboost version 2 test : accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_xgb_ts2, lg_xgb_ts2, f1_xgb_ts2, bas_xgb_ts2))

# save the models 
tv_model = 'tv_transform.sav'
pickle.dump( tv, open( tv_model, 'wb'))

cl_rfc_model = 'rfc_model.sav'
pickle.dump( cl_rfc, open( cl_rfc_model, 'wb'))

xgb_cl2_model = 'xgb_model.sav'
pickle.dump( xgb_cl2, open( xgb_cl2_model, 'wb'))

#  write the results in a txt 
with open(...\\'results.txt', 'a') as f:
    f.write('\n results  \n')
    f.write('\n random forest \n the pest params are: \n')
    f.write( json.dumps(best_values_rfc))
    w1 = "\n random forest test : accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_rfc, lg_rfc, f1_rfc, bas_rfc)
    f.write( w1)
    w2 = " \n CalibratedClassifier test: accuracy%.4f, loss %.4f, f1 %.4f, bca %.4f " %(ac_cl, lg_cl, f1_cl, bas_cl)
    f.write( w2 )
    f.write(" \n xgboost classifier \n best params: \n")
    f.write( json.dumps(best_values_xgb))
    w3 = "\n xgboost version 2 test : accuracy %.4f, loss %.4f, f1 %.4f,  balance accuracy %.4f  " %(ac_xgb_ts2, lg_xgb_ts2, f1_xgb_ts2, bas_xgb_ts2)
    f.write(w3 )

print("end")
