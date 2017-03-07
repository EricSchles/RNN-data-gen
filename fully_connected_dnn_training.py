mport pandas as pd
import numpy as np
np.random.seed(1983)
from pandas import DataFrame
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import sys
import os
import time
import random

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.callbacks import EarlyStopping
# from keras.models import model_from_json

from sklearn.externals import joblib

import matplotlib.pyplot as plt

data_path = '/Users/redaal-bahrani/Documents/Research/SEER/SEER_1973_2010_TEXTDATA/yr1973_2010.seer9.csv'
cutoff_year = 2010
data = pd.read_csv(data_path, dtype=object)
data = data[(data['SITEO2V'].str.contains('C18|C260')) & (data['REC_NO'] == '01')]
data = (data[data['DX_CONF'] != '9'])
data['SITEO2V'].replace(regex=True,inplace=True,to_replace=r'\C',value=r'')

months_survived = data['SRV_TIME_MON']
months_survived = months_survived.astype(int)

data['1_yr_srv'] = months_survived.apply(lambda x: '1' if x >= 12 else '0')

data = data[(~data['CS_SIZE'].isnull()) | (~data['EOD10_SZ'].isnull())]
data['EOD10_SZ'] = data['EOD10_SZ'].fillna(-1000)
data['CS_SIZE'] = data['CS_SIZE'].fillna(-1000)
data['EOD10_SZ'] = data['EOD10_SZ'].astype(int)
data['CS_SIZE'] = data['CS_SIZE'].astype(int)
data['EOD10_SZ'] = data['EOD10_SZ'].apply(lambda x: x/10.0 + 1000 if x < 997 and x > 0 else x + 1000)
data['CS_SIZE'] = data['CS_SIZE'].apply(lambda x: x/10.0 + 1000 if x < 988 and x >= 1 else x + 1000)
data['tumor_size'] = data['EOD10_SZ'] + data['CS_SIZE'] - 1000
data['tumor_size'] = data['tumor_size'].round(2)
data['EOD10_PN'] = data['EOD10_PN'].astype(int)
data['positive_nodes'] = data[(data['EOD10_PN'] > 0) & (data['EOD10_PN'] < 89)]['EOD10_PN']
data['AGE_DX'] = data['AGE_DX'].astype(int)


print(data.columns)
cutoff_year = 2010
year = 1

data = data[data['DATE_yr'].astype(int) <= cutoff_year-year]

att_list = ['1_yr_srv', 'MAR_STAT', 'RACE', 'YR_BRTH', 'PLC_BRTH', 'HISTO3V', 'GRADE', 'DX_CONF', 'EOD10_EX', 'EOD10_ND', 'EOD10_NE', 'SURGPRIM', 'NO_SURG', 'tumor_size', 'positive_nodes', 'AGE_DX']
str_list = ['1_yr_srv', 'MAR_STAT', 'RACE', 'PLC_BRTH', 'GRADE', 'DX_CONF', 'EOD10_EX', 'EOD10_ND', 'EOD10_NE', 'SURGPRIM', 'NO_SURG']
num_list = ['YR_BRTH', 'tumor_size', 'positive_nodes', 'AGE_DX']
data[str_list + num_list] = data[str_list + num_list].fillna(9999)

data = data[str_list]


data_1_yr_srv_ = data_1_yr_srv
for column in list(data_1_yr_srv.columns):
    data_1_yr_srv_[column] = column + '_' + data_1_yr_srv[column].astype(str)


for column in list(data.columns):
    data[column] = column + '_' + data[column].astype(str)

file = open('feature_paragraph_1_yr_no_srv.txt', 'w') 

for i in range(len(data_1_yr_srv_)):
    # print(list(data.iloc[i]))
    # print(' '.join(list(data.iloc[i])))
    file.write(' '.join(list(data_1_yr_srv_.iloc[i])) + ' ')

file.close() 



import pandas as pd
import numpy as np
np.random.seed(1983)
from pandas import DataFrame
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import sys
import os
import time
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

from sklearn.externals import joblib

import matplotlib.pyplot as plt

enc = OneHotEncoder()
enc.fit(data[['MAR_STAT', 'RACE', 'PLC_BRTH', 'GRADE', 'DX_CONF', 'EOD10_EX', 'EOD10_ND', 'EOD10_NE', 'SURGPRIM', 'NO_SURG']])

X = enc.transform(data[['MAR_STAT', 'RACE', 'PLC_BRTH', 'GRADE', 'DX_CONF', 'EOD10_EX', 'EOD10_ND', 'EOD10_NE', 'SURGPRIM', 'NO_SURG']]).toarray()
y = data['1_yr_srv'].as_matrix()

def train_test(random_state=None):
    split_percentage = 0.20
    sss = StratifiedShuffleSplit(y, 1, test_size=split_percentage, random_state=random_state)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = train_test(random_state=12434)
y_train = pd.get_dummies(y_train, sparse=True).as_matrix()
y_test = pd.get_dummies(y_test, sparse=True).as_matrix()

nb_classes = 2
nb_epoch = 10
batch_size = 128
print('Building model...')
model = Sequential()
model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')
history = model.fit(
    X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, 
                    verbose=1, validation_split=0.30, callbacks=[early_stopping])


score = model.evaluate(
    X_test, y_test, batch_size=batch_size, verbose=1)


print('Test score:', score[0])
print('Test accuracy:', score[1])

pred = model.predict_proba(X_test).transpose()[1].round(2)
auc = roc_auc_score(y_test.transpose()[1], pred)
accuracy = accuracy_score(y_test.transpose()[1], pred.round(0))
print()
print('auc: ' + str(auc))
print('accuracy: ' + str(accuracy))

accuracy = accuracy_score(y_test.transpose()[1], pred.round(0))
precision = precision_score(y_test.transpose()[1], pred.round(0))
recall = recall_score(y_test.transpose()[1], pred.round(0))
f1 = f1_score(y_test.transpose()[1], pred.round(0))
area_under_curve = roc_auc_score(y_test.transpose()[1], pred)
cm  = confusion_matrix(y_test.transpose()[1], pred.round(0))
cr = classification_report(y_test.transpose()[1], pred.round(0))

print('accuracy', accuracy)
print('precision', precision)
print('recall', recall)
print('f1', f1)
print('auc of roc', area_under_curve)
print(cm)
print(cr)
ppv = (cm[1][1])*1.0/(cm[1][1]+cm[1][0])*100
print('ppv', ppv)
npv = (cm[0][0])*1.0/(cm[0][0]+cm[0][1])*100
print('npv', npv)
sensitivity = (cm[1][1])*1.0/(cm[1][1]+cm[0][1])*100
print('sensitivity', sensitivity)
specificity = (cm[0][0])*1.0/(cm[0][0]+cm[1][0])*100
print('specificity', specificity)
print('\n')



## code to produce original format of the data from the output of the encoder-decoder
rec_gen = []
for rec in rec_list:
    tmp = {}
    features = rec.split()
    print(features)
    for feature in features:
        print(feature)
        ind = [index for index, value in enumerate(feature) if value == '_'][-1]
        print(feature[0:ind])
        print(feature[ind+1:len(feature)])
        tmp[feature[0:ind]] = feature[ind+1:len(feature)]
    rec_gen.append(tmp)

print(rec_gen)
pd.DataFrame(rec_gen)