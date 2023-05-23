'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 2
'''

class BreadPredictor:
  def __init__(self):
    from sklearn.preprocessing import MinMaxScaler
    self.scalerX = MinMaxScaler()
    self.scalerY = MinMaxScaler()

  def load_data(self):    
    '''
    This method loads the required datasets for 
    the bread price prediction and puts them
    all into dataframes.
    '''
    import pandas as pd
    BREAD_PRICE = pd.read_csv("datasets/monthly_bread.csv")
    GDP = pd.read_csv("datasets/monthly_gdp.csv")
    IMPORTS = pd.read_csv("datasets/monthly_imports.csv")
    EXPORTS = pd.read_csv("datasets/monthly_exports.csv")
    INFLATION = pd.read_csv("datasets/monthly_inflation.csv")
    UNEMPLOYMENT = pd.read_csv("datasets/monthly_unemployment.csv")
    WAGE = pd.read_csv("datasets/monthly_wage.csv")
    return BREAD_PRICE, GDP, IMPORTS, INFLATION, UNEMPLOYMENT, WAGE, EXPORTS
  
  def preprocess_data(self, BREAD_PRICE, GDP, IMPORTS, INFLATION, UNEMPLOYMENT, WAGE, EXPORTS):
    '''
    This method preprocesses the data for training 
    the bread price prediction model by splitting
    the datasets.
    '''
    from sklearn.model_selection import train_test_split
    import numpy as np
    df_X = np.stack((np.array(GDP['GDP'].values), np.array(IMPORTS['IMPORTS'].values), np.array(EXPORTS['EXPORTS'].values),
      np.array(INFLATION['INFLATION'].values), np.array(WAGE['WAGE'].values), np.array(UNEMPLOYMENT['UNEMPLOYMENT'].values)), axis=1)
    df_Y = BREAD_PRICE['BREAD'].values.reshape(-1,1)

    df_X, df_Y = self.scalerX.fit_transform(df_X), self.scalerY.fit_transform(df_Y)
    print(f"Training Dataset: {df_X.shape}")
    print(f"Label Dataset: {df_Y.shape}")
    xtr, xtt, ytr, ytt = train_test_split(df_X, df_Y, test_size=0.1)
    return xtr, xtt, ytr, ytt

  def build_model(self):
    '''
    This method builds the bread price prediction 
    model using Keras, which is a part of the 
    machine learning library TensorFlow.
    '''
    from keras import layers
    from keras.models import Sequential
    model = Sequential()
    model.add(layers.Dense(64, input_dim=6))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Output
    model.compile(loss="mean_squared_error", optimizer="Adam")
    return model

  def fit_save_model(self, model, xtr, ytr, xtt, ytt, model_name):
    '''
    This method trains the bread price prediction model 
    and saves it to a file.
    '''
    from keras.callbacks import EarlyStopping
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode="auto", restore_best_weights=True)
    model.fit(xtr, ytr, validation_data=(xtt, ytt), verbose=2, epochs=200, callbacks=[monitor])
    model.save(model_name)

  def evaluate_model(self, model, xtt, ytt, model_name):
    '''
    This method evaluates the bread price prediction 
    model and displays the evaluation metrics to the
    console.
    '''
    import sklearn.metrics as sm
    import matplotlib.pyplot as plt
    from keras import models
    model = models.load_model(model_name)
    ytt_pred = model.predict(xtt)
    print("Mean absolute error =", round(sm.mean_absolute_error(ytt, ytt_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(ytt, ytt_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(ytt, ytt_pred), 2))
    print(f"R2 Value: {round(sm.r2_score(ytt, ytt_pred), 2)}")

    plt.plot(ytt)
    plt.plot(ytt_pred)
    plt.savefig("MLplots/ModelAccuracy.png")

  def test_future_values(self, gdp, imports, exports, inflation, wage, unemployment, model_name):
    '''
    This method takes in the future values of the
    inputs given and predicts the label based
    on the given features.
    '''
    from keras import models
    x = [[gdp, imports, exports, inflation, wage, unemployment]]
    x = self.scalerX.transform(x)

    model = models.load_model(model_name)
    prediction = model.predict(x)
    print(f"Prediction: {self.scalerY.inverse_transform(prediction)[0][0]}")
    return self.scalerY.inverse_transform(prediction)[0][0]

if __name__ == "__main__":
  # Loading and preprocessing data
  predictor = BreadPredictor()
  BREAD_PRICE, GDP, IMPORTS, INFLATION, UNEMPLOYMENT, WAGE, EXPORTS = predictor.load_data()
  x_train, x_test, y_train, y_test = predictor.preprocess_data(BREAD_PRICE, GDP, IMPORTS, INFLATION, UNEMPLOYMENT, WAGE, EXPORTS)

  # try to visualize your data here
  import matplotlib.pyplot as plt
  plt.style.use("seaborn-darkgrid")
  fig, ax = plt.subplots(3,2)
  fig.tight_layout(h_pad=2)
  ax[0,0].plot(GDP["GDP"],color="red")
  ax[0,0].set_title("GDP")
  ax[0,1].plot(IMPORTS["IMPORTS"], color='green')
  ax[0,1].set_title("Imports")
  ax[1,0].plot(EXPORTS["EXPORTS"], color='blue')
  ax[1,0].set_title("Exports")
  ax[1,1].plot(INFLATION["INFLATION"], color='orange')
  ax[1,1].set_title("Inflation")
  ax[2,0].plot(WAGE["WAGE"], color='pink')
  ax[2,0].set_title("Minimum Wage")
  ax[2,1].plot(UNEMPLOYMENT["UNEMPLOYMENT"], color='yellow')
  ax[2,1].set_title("Unemployment")
  plt.subplots_adjust(top=0.85)
  plt.savefig("MLplots/ShowFeatures.png")

  # building and fitting the model
  model = predictor.build_model()
  predictor.fit_save_model(model, x_train, y_train, x_test, y_test, "Model.h5")
  predictor.evaluate_model(model, x_test, y_test, "Model.h5")

  # Evaluate future values
  gdp_future = []
  for i in range(50):
    prediction = predictor.test_future_values(gdp=GDP["GDP"].iloc[-1].tolist()+i,imports=IMPORTS["IMPORTS"].iloc[-1].tolist(),exports=EXPORTS["EXPORTS"].iloc[-1].tolist(),
      inflation=INFLATION["INFLATION"].iloc[-1].tolist(),wage=WAGE["WAGE"].iloc[-1].tolist(),unemployment=UNEMPLOYMENT["UNEMPLOYMENT"].iloc[-1].tolist(), model_name="Model.h5")
    gdp_future.append(prediction)

  inflation_future = []
  for i in range(50):
    prediction = predictor.test_future_values(gdp=GDP["GDP"].iloc[-1].tolist(),imports=IMPORTS["IMPORTS"].iloc[-1].tolist(),exports=EXPORTS["EXPORTS"].iloc[-1].tolist(),
      inflation=INFLATION["INFLATION"].iloc[-1].tolist()+i,wage=WAGE["WAGE"].iloc[-1].tolist(),unemployment=UNEMPLOYMENT["UNEMPLOYMENT"].iloc[-1].tolist(), model_name="Model.h5")
    inflation_future.append(prediction)

  wage_future = []
  for i in range(50):
    prediction = predictor.test_future_values(gdp=GDP["GDP"].iloc[-1].tolist(),imports=IMPORTS["IMPORTS"].iloc[-1].tolist(),exports=EXPORTS["EXPORTS"].iloc[-1].tolist(),
      inflation=INFLATION["INFLATION"].iloc[-1].tolist(),wage=WAGE["WAGE"].iloc[-1].tolist()+i,unemployment=UNEMPLOYMENT["UNEMPLOYMENT"].iloc[-1].tolist(), model_name="Model.h5")
    wage_future.append(prediction)

  unemployment_future = []
  for i in range(50):
    prediction = predictor.test_future_values(gdp=GDP["GDP"].iloc[-1].tolist(),imports=IMPORTS["IMPORTS"].iloc[-1].tolist(),exports=EXPORTS["EXPORTS"].iloc[-1].tolist(),
      inflation=INFLATION["INFLATION"].iloc[-1].tolist(),wage=WAGE["WAGE"].iloc[-1].tolist(),unemployment=UNEMPLOYMENT["UNEMPLOYMENT"].iloc[-1].tolist()+i, model_name="Model.h5")
    unemployment_future.append(prediction)

  imports_future = []
  for i in range(50):
    prediction = predictor.test_future_values(gdp=GDP["GDP"].iloc[-1].tolist(),imports=IMPORTS["IMPORTS"].iloc[-1].tolist()+(i*10000000000),exports=EXPORTS["EXPORTS"].iloc[-1].tolist(),
      inflation=INFLATION["INFLATION"].iloc[-1].tolist(),wage=WAGE["WAGE"].iloc[-1].tolist(),unemployment=UNEMPLOYMENT["UNEMPLOYMENT"].iloc[-1].tolist()+i, model_name="Model.h5")
    imports_future.append(prediction)

  exports_future = []
  for i in range(50):
    prediction = predictor.test_future_values(gdp=GDP["GDP"].iloc[-1].tolist(),imports=IMPORTS["IMPORTS"].iloc[-1].tolist(),exports=EXPORTS["EXPORTS"].iloc[-1].tolist()+(i*10000000000),
      inflation=INFLATION["INFLATION"].iloc[-1].tolist(),wage=WAGE["WAGE"].iloc[-1].tolist(),unemployment=UNEMPLOYMENT["UNEMPLOYMENT"].iloc[-1].tolist()+i, model_name="Model.h5")
    exports_future.append(prediction)
  
  # Plotting Data
  fig, ax = plt.subplots(3,2)
  fig.tight_layout(h_pad=2)
  ax[0,0].plot(gdp_future,color="red")
  ax[0,0].set_title("GDP Change")
  ax[0,1].plot(imports_future, color='green')
  ax[0,1].set_title("Imports Change")
  ax[1,0].plot(exports_future, color='blue')
  ax[1,0].set_title("Exports Change")
  ax[1,1].plot(inflation_future, color='orange')
  ax[1,1].set_title("Inflation Change")
  ax[2,0].plot(wage_future, color='pink')
  ax[2,0].set_title("Minimum Wage Change")
  ax[2,1].plot(unemployment_future, color='yellow')
  ax[2,1].set_title("Unemployment Change")
  plt.subplots_adjust(top=0.85)
  plt.savefig("MLplots/PredictedPriceChange.png")



  
