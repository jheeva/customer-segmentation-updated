# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:26:39 2022

@author: End User
"""


#%%modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from cust_segmentation import EDA, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
#from tensorflow.keras.utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix,classification_report
import pickle
import seaborn as sns

import missingno as msno

#static/constants
DATA_PATH = os.path.join(os.getcwd(), "data","train.csv")
PATH_LOGS = os.path.join(os.getcwd(), 'history')
log_files = os.path.join(PATH_LOGS, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models','model.h5')
EDA_PATH_SAVE=os.path.join(os.getcwd(),"saved_models",'eda_scaler.pkl')
#%%data loading
df = pd.read_csv(DATA_PATH)

#%%data inspection using graph
df.info()

#gender against your segemntation(A,B,C,D)
sns.countplot(df['Profession'],hue=df['Segmentation'])

# profesion vs segmentation
sns.countplot(df["Profession"], hue=df["Segmentation"])

# Ever_married vs segmentation
sns.countplot(df["Ever_Married"], hue=df["Segmentation"])

df.groupby(["Segmentation", "Ever_Married", "Gender"]).agg({"Segmentation":"count"}).plot(kind="bar")


#data cleaning
msno.matrix(df)


X_raw = df.iloc[:,1:10] # features only

enc = EDA()
# label encoder
X_clean = enc.label_encoder(X_raw)

# fill the NaN
X_imputed = enc.simple_imputer(X_clean)
#%% Feature Scalling

X_scaled = enc.feature_scaling(X_imputed)

#%%build and train the model

# convert target to one_hot encoding
y_raw = np.expand_dims(df["Segmentation"], axis=-1)
y_encoded = enc.one_hot_encoding(y_raw)

# assign the features and target
X = X_scaled
y = y_encoded

# splitting (X_test,y_test,X_train,y_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=29)

#%%model
model = Model()

# fit the data
clf = model.neural_network(nb_features=9, nodes1=64, nodes2=86,
                       activation1="relu", nb_target=4, 
                       out_activation="softmax")

#tensorboard and early stopping
tensorboard_callback=TensorBoard(log_dir=log_files,histogram_freq=1)
es_callback=EarlyStopping(monitor='val_loss',patience=3)

callbacks=[tensorboard_callback,es_callback]


#compile,train and evaluate model
clf.compile(optimizer="adam",
            loss="categorical_crossentropy",
            metrics="acc")


clf.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test),
             callbacks=[es_callback, tensorboard_callback])


#%% model saving
clf.save(MODEL_SAVE_PATH)

#%%evaluation
y_pred=clf.predict(X_test)


cm=confusion_matrix(np.argmax(np.array(y_test),axis=-1),
                    np.argmax(y_pred,axis=-1),labels=[i for i in range(8)])


cr=classification_report(np.argmax(np.array(y_test),axis=1),
                    np.argmax(y_pred,axis=1))

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=(y_test))

disp.plot(cmap=plt.cm.Blues)
plt.show()






