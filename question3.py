'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 3
'''

#imports
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Sequential
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from keras import models
from keras.callbacks import EarlyStopping

class BreadPredictor:
  def __init__(self):
    self.scalerX = MinMaxScaler()
    self.scalerY = MinMaxScaler()

  def load_data(self):    
    BREAD_PRICE = pd.read_csv("datasets/monthly_bread.csv")
    GDP = pd.read_csv("datasets/monthly_gdp.csv")
    IMPORTS = pd.read_csv("datasets/monthly_imports.csv")
    EXPORTS = pd.read_csv("datasets/monthly_exports.csv")
    INFLATION = pd.read_csv("datasets/monthly_inflation.csv")
    UNEMPLOYMENT = pd.read_csv("datasets/monthly_unemployment.csv")
    WAGE = pd.read_csv("datasets/monthly_wage.csv")

    df_X = np.stack((np.array(GDP['GDP'].values), np.array(IMPORTS['IMPORTS'].values), np.array(EXPORTS['EXPORTS'].values),
      np.array(INFLATION['INFLATION'].values), np.array(WAGE['WAGE'].values), np.array(UNEMPLOYMENT['UNEMPLOYMENT'].values)), axis=1)
    df_Y = BREAD_PRICE['BREAD'].values.reshape(-1,1)

    df_X, df_Y = self.scalerX.fit_transform(df_X), self.scalerY.fit_transform(df_Y)
    print(f"Training Dataset: {df_X.shape}")
    print(f"Label Dataset: {df_Y.shape}")
    return df_X, df_Y
  
  def convert_dataframe(self, x_data, y_data):
    dfX = pd.DataFrame(x_data, columns=["GDP", "Imports", "Exports", "Inflation", "Wage", "Unemployment"])
    dfY = pd.DataFrame(y_data, columns=["Bread"])
    return dfX, dfY
  
  def split_data(self, df_X, df_Y):
    xtr, xtt, ytr, ytt = train_test_split(df_X, df_Y, test_size=0.9)
    return xtr, xtt, ytr, ytt

  def build_model(self):
    model = Sequential()
    model.add(layers.Dense(64, input_dim=6))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Output
    model.compile(loss="mean_squared_error", optimizer="Adam")
    return model

  def fit_save_model(self, model, xtr, ytr, xtt, ytt, model_name):
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode="auto", restore_best_weights=True)
    model.fit(xtr, ytr, validation_data=(xtt, ytt), verbose=2, epochs=200, callbacks=[monitor])
    model.save(model_name)

  def evaluate_model(self, model, xtt, ytt, model_name):
    model = models.load_model(model_name)
    ytt_pred = model.predict(xtt)
    print("Mean absolute error =", round(sm.mean_absolute_error(ytt, ytt_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(ytt, ytt_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(ytt, ytt_pred), 2))
    print(f"R2 Value: {round(sm.r2_score(ytt, ytt_pred), 2)}")

    plt.plot(ytt)
    plt.plot(ytt_pred)
    plt.savefig("ModelAccuracy.png")

  def test_future_values(self, gdp, imports, exports, inflation, wage, unemployment, model_name):
    from keras import models
    x = [[gdp, imports, exports, inflation, wage, unemployment]]
    x = self.scalerX.transform(x)

    model = models.load_model(model_name)
    prediction = model.predict(x)
    print(f"Prediction: {self.scalerY.inverse_transform(prediction)[0][0]}")

if __name__ == "__main__":
  # Loading and preprocessing data
  predictor = BreadPredictor()
  x_data, y_data = predictor.load_data()
  x_train, x_test, y_train, y_test = predictor.split_data(x_data, y_data)
  df_x, df_y = predictor.convert_dataframe(x_data, y_data)

  # try to visualize your data here
  plt.plot(df_x["GDP"], color='red')
  plt.plot(df_x["Imports"], color='green')
  plt.plot(df_x["Exports"], color='blue')
  plt.plot(df_x["Inflation"], color='orange')
  plt.plot(df_x["Wage"], color='pink')
  plt.plot(df_x["Unemployment"], color='yellow')
  plt.title("Features to Predict Bread Price")
  plt.legend(["GDP", "Imports", "Exports", "Inflation", "Wage", "Unemployment"])
  plt.savefig("FeaturesPlotted.png")

  # building and fitting the model
  model = predictor.build_model()
  predictor.fit_save_model(model, x_train, y_train, x_test, y_test, "Model.h5")
  predictor.evaluate_model(model, x_test, y_test, "Model.h5")

  # Evaluate future values
  predictor.test_future_values(gdp=100,imports=5,exports=5,inflation=5,wage=5,unemployment=5, model_name="Model.h5")
