# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:37:35 2022

@author: End User
"""


import pandas as pd
import numpy as np

import os
from cust_segmentation import EDA
from tensorflow.keras.models import load_model

# static/constants
DATA_PATH = os.path.join(os.getcwd(), "data","new_customers.csv")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models','model.h5')

#%% EDA
df_new = pd.read_csv(DATA_PATH)

X_raw = df_new.iloc[:,1:10]

enc = EDA()


'''LABEL ENCODER'''
X_clean = enc.label_encoder(X_raw)

''' fill the NaN'''
X_processed = enc.simple_imputer(X_clean)

# scaling
X_scaled = enc.feature_scaling(X_processed)

''' Load Model(for testing)'''
loaded_model = load_model(MODEL_SAVE_PATH)
loaded_model.summary()


prediction =loaded_model.predict(X_scaled)
y_pred = np.argmax(prediction, axis = 1)




# concat in dataframe
X_New = pd.DataFrame(X_scaled)
y_pred = pd.DataFrame(y_pred)

# full completed data
new_form_data = pd.concat([X_New, y_pred], axis = 1)

