#%% Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,classification_report

#%% Constant
DATA_PATH = os.path.join(os.getcwd(), "Data","train.csv")


#%% EDA
class EDA():
    
    def __init__(self):
        pass
    
    def label_encoder(self,df):
        encoded = LabelEncoder()
        df["Gender"] = encoded.fit_transform(df["Gender"])
        df["Ever_Married"] = encoded.fit_transform(df["Ever_Married"])
        df["Graduated"] = encoded.fit_transform(df["Graduated"])
        df["Profession"] = encoded.fit_transform(df["Profession"])
        df["Spending_Score"] = encoded.fit_transform(df["Spending_Score"])
        df["Var_1"] = encoded.fit_transform(df["Var_1"])
        
        return df
        
    def simple_imputer(self,X):
        imputed = SimpleImputer(strategy="median")
        
        return imputed.fit_transform(X)
    
    def feature_scaling(self, X):
        mms = MinMaxScaler()
        
        return mms.fit_transform(X)
    
    def one_hot_encoding(self, df):
        ohe = OneHotEncoder(sparse=False)
        
        return ohe.fit_transform(df)
        
class Model():
    
    def neural_network(self, nb_features, nodes1, nodes2, activation1, 
                       nb_target, out_activation):
        
        model = Sequential(name=("Customer_Segmentation"))
        
        model.add(Input(shape=(nb_features), name="Input_layer"))
        model.add(Dense(nodes1, activation=activation1, name="Input_layer_1"))
        model.add(Dropout(0.2))
        model.add(Dense(nodes2, activation=activation1, name="Input_layer_2"))
        model.add(Dense(nodes1, activation=activation1, name="Input_layer_3"))
        model.add(Dense(nb_target, activation=out_activation, name="Output_layer"))
        print(model.summary())
        
        return model
    
class Evaluation():
    
    def model_eval(self, y_true, y_pred):
    
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)
        print(cr)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=np.unique(y_true))
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
       
    
    
#%% main

if __name__ == "__main__":
    
    df = pd.read_csv(DATA_PATH)
    X_raw = df.iloc[:,1:10] # features only
    
    enc = EDA()
    # label encoder
    X_clean = enc.label_encoder(X_raw)
    
    # fill the NaN
    X_processed = enc.simple_imputer(X_clean)
    
    # scale the features
    X_scaled = enc.feature_scaling(X_processed)
    
    #  one_hot encoding
    y_raw = np.expand_dims(df["Segmentation"], axis=-1)
    y_encoded = enc.one_hot_encoding(y_raw)
    
    
    
    X = X_scaled
    y = y_encoded
    
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # model
    model = Model()
    
    # data fitting
    clf = model.neural_network(nb_features=9, nodes1=64, nodes2=32
                           ,activation1="relu", nb_target=4, 
                           out_activation="softmax")
