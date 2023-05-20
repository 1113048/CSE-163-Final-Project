'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 3
'''

# All Importations
import keras
from keras.model import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import sklearn.metrics as sm
import numpy as np
import pandas as pd


def fit_and_predict():
  # Loading CSV Files into Dataframes
  bread = pd.read_csv('datasets/bread.csv')
  gdp = pd.read_csv('datasets/gdp.csv')

  train_dataset = np.stack((np.array(gdp['GEPUCURRENT'].values).reshape(-1,1)))
  label_dataset = bread['bread'].values.reshape(-1,1)

  x_train, x_test, y_train, y_test = train_test_split(train_dataset, label_dataset, test_size=0.1)
  model = Sequential()
    
  print(f'Training Dataset: {train_dataset.shape}')
  print(f'Label Dataset: {label_dataset.shape}')


def main():
  fit_and_predict()

if __name__ == '__main__':
  main()
