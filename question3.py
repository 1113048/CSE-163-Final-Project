'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 3
'''

# All Importations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import sklearn.metrics as sm
import numpy as np
import pandas as pd


def fit_and_predict():
  bread = pd.read_csv("datasets/bread.csv")
  gdp = pd.read_csv("datasets/gdp.csv")

  train_dataset = gdp["GEPUCURRENT"].values.reshape(-1, 1)
  label_dataset = bread["bread"].values.reshape(-1, 1)

  print(f"Training Dataset: {train_dataset.shape}")
  print(f"Label Dataset: {label_dataset.shape}")

  x_train, x_test, y_train, y_test = train_test_split(train_dataset, label_dataset, test_size=0.1)

  model = keras.Sequential()
  model.add(layers.Dense(64, input_dim=x_train.shape[1]))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(32, activation='relu'))
  model.add(layers.Dense(1))  # Output
  model.compile(loss="mean_squared_error", optimizer="Adam")
  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode="auto", restore_best_weights=True)
  model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=1000)

  y_test_pred = model.predict(x_test)
  print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
  print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
  print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
  print(f"R2 Value: {round(sm.r2_score(y_test, y_test_pred), 2)}")


def main():
  fit_and_predict()

if __name__ == '__main__':
  main()
