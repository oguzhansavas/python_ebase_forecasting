# Python-EBASE Forecasting Implementation
# Author: Oguzhan Savas

# libraries
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def get_data(folder, data_name):
    file = folder / data_name
    data = pd.read_csv(file, delimiter=';', decimal=',')
    return data

def add_data(data_to_be_added, file_name, dataframe, folder):
    '''
    Adds the desired additional wind or solar data.
    data_to_be_added: "SOLAR" or "WIND"
    file_name: file name of the data to be added
    dataframe: dataframe which the additional data will be added to
    folder: path of the folder which the data to be added is located
    '''
    data = get_data(folder, file_name)
    data = pd.DataFrame(data)
    data, features_data = data_preprocessing(data)
    dataframe[data_to_be_added] = data.iloc[:,0] * -1
    return dataframe

def data_preprocessing(dataframe):
    '''
    Prepares the dataframe for the forecast.
    dataframe: dataframe to be preprocessed
    '''
    # Get feature names
    features = [col for col in dataframe.columns]
    dict_name = {}
    # Add feature names to dictionary
    for count, feature in enumerate(features):
        if count > 1:
            dict_name[feature] = feature
        elif count == 0:
            dict_name[feature] = 'Date'
        elif count == 1:
            dict_name[feature] = 'Time'
    # Rename columns
    dataframe = dataframe.rename(columns=dict_name)
    # Create datetime column, convert to 'datetime' object
    dataframe['DateTime'] = dataframe['Date'].astype(str) +"-"+ dataframe['Time']
    dataframe = dataframe.drop(['Date', 'Time'], axis=1)
    dataframe = dataframe.iloc[:, [1,0]]
    dataframe['DateTime'] = pd.to_datetime(dataframe['DateTime'], format='%d/%m/%Y-%H:%M')
    # Get all numeric features
    dataframe_num = dataframe.select_dtypes(include=['float', 'int'])
    numeric_features = [col for col in dataframe_num.columns]
    # Make sure all numeric features are float
    for feature in numeric_features:
        dataframe = dataframe.astype({feature:'float'})
    # Set index as DateTime
    dataframe = dataframe.set_index('DateTime')
    # NaN handling
    dataframe[numeric_features] = dataframe[numeric_features].fillna(dataframe.rolling(5,min_periods=1, center=True).mean())
    return dataframe, features    


def divide_data_to_train_test(dataframe, horizon):
    """
    Divides the data to train and test periods based on the given horizon.
    dataframe: data frame to be divided
    horizon: days in the past to be considered
    """
    day_count = (dataframe.shape[0])/24
    if horizon > day_count:
         raise Exception("Number of days to train exceeds the day count!")
    else:
        dataframe = dataframe.reset_index()
        end_train = 24*horizon
        data_test  = dataframe.iloc[-24:, :].set_index('DateTime')
        data_train = dataframe.iloc[-end_train:-24, :].set_index('DateTime')
        #data_train = dataframe.iloc[-end_train:, :].set_index('DateTime')
        #data_test  = dataframe.iloc[-(end_train+24):-end_train, :].set_index('DateTime')
        data_test['Hour'] = data_test.index.hour
        data_train['Hour'] = data_train.index.hour
        data_test['Week Day'] = data_test.index.weekday
        data_train['Week Day'] = data_train.index.weekday
    return data_train, data_test


def create_train_test_sets(data_train, data_test, solar=False, wind=False):
    """
    Creates the train and test sets which the ML model will be trained and tested on.
    data_train: training data to be divided
    data_test: test data to be divided
    solar: True if solar data is also considered
    wind: True if wind data is also considered 
    """
    # Create sets based on provided data
    if solar and not wind:
        X_train, X_test = data_train[['SOLAR', 'Hour', 'Week Day']], data_test[['SOLAR', 'Hour', 'Week Day']]
        y_train, y_test = data_train.drop(['SOLAR', 'Hour', 'Week Day'], axis=1), data_test.drop(['SOLAR', 'Hour', 'Week Day'], axis=1)
    elif wind and not solar:
        X_train, X_test = data_train[['WIND', 'Hour', 'Week Day']], data_test[['WIND', 'Hour', 'Week Day']]
        y_train, y_test = data_train.drop(['WIND', 'Hour', 'Week Day'], axis=1), data_test.drop(['WIND', 'Hour', 'Week Day'], axis=1)
    elif solar and wind:
        X_train, X_test = data_train[['SOLAR', 'WIND', 'Hour', 'Week Day',]], data_test[['SOLAR', 'WIND', 'Hour', 'Week Day']]
        y_train, y_test = data_train.drop(['SOLAR', 'WIND', 'Hour', 'Week Day'], axis=1), data_test.drop(['SOLAR', 'WIND', 'Hour', 'Week Day'], axis=1)
    else:
        X_train, X_test = data_train[['Hour', 'Week Day']], data_test[['Hour', 'Week Day']]
        y_train, y_test = data_train.drop(['Hour', 'Week Day'], axis=1), data_test.drop(['Hour', 'Week Day'], axis=1)
    # Reset index and get rid of the date time column
    X_train, X_test = X_train.reset_index().drop(['DateTime'], axis=1), X_test.reset_index().drop(['DateTime'], axis=1)
    y_train, y_test = y_train.reset_index().drop(['DateTime'], axis=1), y_test.reset_index().drop(['DateTime'], axis=1)
    return X_train, X_test, y_train, y_test

#def get_autocorrelation_of_lags():


def convert_to_supervised(data, n_in=1, n_out=1, dropnan=False, replacenan=True):
    """
    data: Dataframe to be used to create a supervised learning set.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    """
    n_features = data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_features)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
          names += [('var%d(t)' % (j+1)) for j in range(n_features)]
        else:
          names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_features)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    if replacenan:
        agg = agg.fillna(agg.rolling(7, min_periods=1, center=True).mean())
    return agg


def plot_time_series(data_train, data_test):
    fig, ax = plt.subplots(figsize=(12, 4))
    data_train.ALLOCATION.plot(ax=ax, label='train', linewidth=1)
    data_test.ALLOCATION.plot(ax=ax, label='test', linewidth=2, ls='-.')
    ax.set_title('Data to be used')
    ax.legend()
    return ax


def tune_model(X_train, y_train):
    """
    Returns the best model after hyperparameter tuning
    X_train: independent features which the model is trained on
    y_train: dependent features (target) which the model is trained on
    """
    params = {'max_depth': [3, 6, 9, 12],
              'learning_rate': [ 0.05, 0.1, 0.15, 0.2],
              'subsample': np.arange(0.5, 1.0, 0.1),
              'colsample_bytree': np.arange(0.5, 1.0, 0.1),
              'colsample_bylevel': np.arange(0.5, 1.0, 0.1),
              'n_estimators': [100, 200, 300, 400, 500]
              }

    model = xgb.XGBRegressor(objective="reg:squarederror", tree_method='hist')
    reg = RandomizedSearchCV(estimator=model,
                             param_distributions=params,
                             scoring='neg_root_mean_squared_error',
                             n_iter=100,
                             n_jobs=3,
                             verbose=1)

    reg.fit(X_train, y_train)
    best_model = reg.best_estimator_
    return best_model

 

if __name__ == '__main__':

    ##### User input #####
    horizon = 22
    add_solar = True
    add_wind = True
    folder_path = r"C:\Users\oguzhan.savas\OneDrive - Energy21\Documents\Python_Integration\python_ebase_forecasting"
    real_data = "data_electricity.csv"
    ebase_forecast = "ebase_electricity.csv"
    solar_data = "solar_data.csv"
    wind_data = "wind_data.csv"
    ######################

    # Get data
    folder = Path(folder_path)
    usage_data = get_data(folder, real_data)
    df = pd.DataFrame(usage_data)
    df, features = data_preprocessing(df)

    # If the user want to add wind or solar data as well 
    if add_solar:
        df = add_data('SOLAR', solar_data, df, folder)
    if add_wind:
        df = add_data('WIND', wind_data, df, folder)

    # Divide your data as train and test
    data_train, data_test = divide_data_to_train_test(df, horizon)
    # Create train-test sets to be used while model creation
    X_train, X_test, y_train, y_test = create_train_test_sets(data_train, data_test, solar=add_solar, wind=add_wind)

    # Get the necessary amount of lags by checking the autocorrelation

    # Apply lag to target variables
    #y_train_lagged = convert_to_supervised(y_train, n_in=3, n_out=1, dropnan=False, replacenan=True)

    # Hyperparameter tuning
    xgb_model = tune_model(X_train, y_train)

    # Prediction
    preds = xgb_model.predict(X_test)


    # For development (comparison - visualization)
    # Get ebase forecast data
    file = folder / ebase_forecast
    ebase_data = pd.read_csv(file, delimiter=';', decimal=',')
    # Create dataframe
    df_ebase = pd.DataFrame(ebase_data)
    df_ebase, features_ebase = data_preprocessing(df_ebase)

    # Divide ebase data as train and test
    data_train_ebase, data_test_ebase = divide_data_to_train_test(df_ebase, horizon)
    # Create train-test sets (only y_test_ebase is used for visualization and comparison)
    X_train_ebase, X_test_ebase, y_train_ebase, y_test_ebase = create_train_test_sets(data_train_ebase, data_test_ebase, solar=False, wind=False)

    # Plot predictions vs test
    plt.plot(X_test['Hour'], preds, color='r', label='preds')
    plt.plot(X_test['Hour'], y_test, color='g', label='test', alpha=0.8)
    plt.plot(X_test['Hour'], y_test_ebase, color='c', label='ebase', ls='--')
    plt.xlabel("Hours")
    plt.ylabel("Amount")
    plt.title("Predictions vs Observations vs Ebase")
    plt.legend()
    plt.show()

    # Create dataframe for predictions - this data can be sent back to ebase (make it a function)