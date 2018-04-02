import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score

data = pd.read_csv('/Users/xiujiayang/PycharmProjects/DP_compression/train_sample.csv')
data.drop(labels=['attributed_time'], inplace=True, axis=1)
data['_time'] = data['click_time'].str.slice(11,)
data['shijian'] = data['_time'].str.slice(0, 2).astype('int')*3600 + data['_time'].str.slice(3, 5).astype('int')*60 + \
                  data['_time'].str.slice(6,).astype('int')
data['click_time'] = data['click_time'].str.slice(0, 11)
leader_code = LabelEncoder()
data_cat = leader_code.fit_transform(data['click_time'])
cat_encoder = OneHotEncoder()
data_cat_reshaped = data_cat.reshape(-1, 1)
data_cat_1hot = cat_encoder.fit_transform(data_cat_reshaped)
data_date = pd.DataFrame(data_cat_1hot.toarray())
data_reformed = pd.concat([data[['..ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'shijian']],
                           data_date], axis=1)

# Split the data to trainset and testset
split = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
for train_index, test_index in split.split(data_reformed, data_reformed['is_attributed']):
    train = data_reformed.loc[train_index]
    test = data_reformed.loc[test_index]

train_data = train.loc[:, train.columns != 'is_attributed']
train_label = train.loc[:, train.columns == 'is_attributed']
test_data = test.loc[:, test.columns != 'is_attributed']
test_label = test.loc[:, test.columns == 'is_attributed']
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

# Deal with imbalanced data
weights = (len(train_label)-np.sum(train_label))/np.sum(train_label)

# I use Xgboost classifier and start parameter tuning
# Tuning max_depth, min_child_weight
'''
param_test1 = {
    'max_depth': range(3, 12, 2),
    'min_child_weight': range(1, 6, 2)
}

gsearch1 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights, 
                                                learning_rate=0.1,
                                                n_estimators=150,
                                                max_depth=5,
                                                min_child_weight=1,
                                                gamma=0,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, seed=40),
                        param_grid=param_test1,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch1.fit(train_data, train_label.ravel())
print(gsearch1.best_params_, gsearch1.best_score_) 
# optimal parameter: {'max_depth': 3, 'min_child_weight': 5} 0.9590893294239716

xgb1 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=150,
                     max_depth=3,
                     min_child_weight=5,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic', nthread=4, seed=40)
xgb1.fit(train_data, train_label.ravel())
predictions1= xgb1.predict(test_data)
accuracy1 = accuracy_score(test_label, predictions1)
recall1 = recall_score(test_label, predictions1)
f1_1 = f1_score(test_label, predictions1)
auc1 = roc_auc_score(test_label, predictions1)
auc_train1 = roc_auc_score(train_label, xgb1.predict(train_data))
print(accuracy1, recall1, f1_1, auc1, auc_train1)
# get: 0.9834333333333334 0.8529411764705882 0.18923327895595432 0.9183354819944817 0.9918457639495426

# Tuning gamma
param_test2 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}
gsearch2 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=150,
                                                max_depth=3,
                                                min_child_weight=5,
                                                gamma=0,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, seed=40),
                        param_grid=param_test2,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)
gsearch2.fit(train_data, train_label.ravel())
print(gsearch2.best_params_, gsearch2.best_score_) 
# optimal parameter: {'gamma': 0.2} 0.9592708748559691

xgb2 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=150,
                     max_depth=3,
                     min_child_weight=5,
                     gamma=0.2,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic', nthread=4, seed=40)
xgb2.fit(train_data, train_label.ravel())
predictions2 = xgb2.predict(test_data)
accuracy2 = accuracy_score(test_label, predictions2)
recall2 = recall_score(test_label, predictions2)
f1_2 = f1_score(test_label, predictions2)
auc2 = roc_auc_score(test_label, predictions2)
auc_train2 = roc_auc_score(train_label, xgb2.predict(train_data))
print(accuracy2, recall2, f1_2, auc2, auc_train2)
# get: 0.9834333333333334 0.8529411764705882 0.18923327895595432 0.9183354819944817 0.9918457639495426

# Tune subsample and colsample_bytree
param_test3 = {
    'subsample': [i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}
gsearch3 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=150,
                                                max_depth=3,
                                                min_child_weight=5,
                                                gamma=0.2,
                                                subsample=0.8,
                                                colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, seed=40),
                        param_grid=param_test3,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)

gsearch3.fit(train_data, train_label.ravel())
print(gsearch3.best_params_, gsearch3.best_score_) 
# optimal parameter: {'colsample_bytree': 0.7, 'subsample': 0.7} 0.9620979137690469

xgb3 = XGBClassifier(scale_pos_weight=weights,
                     learning_rate=0.1,
                     n_estimators=150,
                     max_depth=3,
                     min_child_weight=5,
                     gamma=0.2,
                     subsample=0.7,
                     colsample_bytree=0.7,
                     objective='binary:logistic', nthread=4, seed=40)
xgb3.fit(train_data, train_label.ravel())
predictions3 = xgb3.predict(test_data)
accuracy3 = accuracy_score(test_label, predictions3)
recall3 = recall_score(test_label, predictions3)
f1_3 = f1_score(test_label, predictions3)
auc3 = roc_auc_score(test_label, predictions3)
auc_train3 = roc_auc_score(train_label, xgb3.predict(train_data))
print(accuracy3, recall3, f1_3, auc3, auc_train3)
# get: 0.9842333333333333 0.8970588235294118 0.20504201680672268 0.940745100659534 0.9920533783880529


# Tuning learning_rate
param_test5 = {
 'learning_rate': [1, 0.5, 0.25, 0.1, 0.075, 0.05, 0.01, 0.005, 0.001]
}
gsearch5 = GridSearchCV(estimator=XGBClassifier(scale_pos_weight=weights,
                                                learning_rate=0.1,
                                                n_estimators=150,
                                                max_depth=3,
                                                min_child_weight=5,
                                                gamma=0.2,
                                                subsample=0.7,
                                                colsample_bytree=0.7,
                                                objective='binary:logistic', nthread=4, seed=40),
                        param_grid=param_test5,
                        scoring='roc_auc',
                        n_jobs=4,
                        iid=False,
                        cv=5)

gsearch5.fit(train_data, train_label.ravel())
print(gsearch5.best_params_, gsearch5.best_score_)
# optimal parameter: {'learning_rate': 0.1} 0.9620979137690469
'''
model = XGBClassifier(scale_pos_weight=weights,
                      learning_rate=0.1,
                      n_estimators=150,
                      max_depth=3,
                      min_child_weight=5,
                      gamma=0.2,
                      subsample=0.7,
                      colsample_bytree=0.7,
                      objective='binary:logistic', nthread=4, seed=40)
model.fit(train_data, train_label.ravel())
predictions = model.predict(test_data)
accuracy = accuracy_score(test_label, predictions)
recall = recall_score(test_label, predictions)
f1 = f1_score(test_label, predictions)
auc = roc_auc_score(test_label, predictions)
auc_train = roc_auc_score(train_label, model.predict(train_data))
print (accuracy, recall, f1, auc, auc_train)
# get final: 0.9842333333333333 0.8970588235294118 0.20504201680672268 0.940745100659534 0.9920533783880529
print(model)
'''
final model: XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bytree=0.7, gamma=0.3, learning_rate=0.011,
             max_delta_step=0, max_depth=9, min_child_weight=5, missing=None,
             n_estimators=100, n_jobs=1, nthread=None,
             objective='binary:logistic', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=439.251572327044, seed=None,
             silent=True, subsample=0.9)
'''
