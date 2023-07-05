import subprocess
def install_library(library_name):
    subprocess.check_call(['pip', 'install', library_name])
    
# Usage
install_library('feature_engine')

from operator import index
from statsmodels.tsa.seasonal import STL
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import pacf
from numpy.fft import rfft, irfft, rfftfreq
import statsmodels as sm


def constantIdentifier(df,input_cols,constant_features):
    for col_name in input_cols:
            unique_values = df.select(col_name).distinct().count()
            if unique_values == 1:
                constant_features.append(col_name)

    return constant_features





def missing_handler(df):
    num_cols = df.select_dtypes(include=['number'])
    for col in num_cols:
        df2 = df.interpolate(method="linear")
        Decomposed = STL(df2[col],robust=True,period = 12).fit()  
        df[col] = df[col] - Decomposed.seasonal
        df[col] = df[col].interpolate(method="linear")
        df[col] = df[col] + Decomposed.seasonal
    return df






# learn the outlier limits
def outlier_handler(X,cappers):
    for col in X.select_dtypes(include='number'):
        decomposed = STL(X[col], robust=True,period = 12).fit()
        residuals = decomposed.resid
        capper = Winsorizer(capping_method="iqr", tail='both', fold=1.5)
        capper.fit(pd.DataFrame(residuals))
        cappers[col] = capper
    return cappers





# apply the outlers limits
def outlier_capper(X,cappers):
    for col in X.select_dtypes(include='number'):
        decomposed = STL(X[col], robust=True,period = 12).fit()
        residuals = decomposed.resid
        capper = cappers[col]
        residuals = capper.transform(pd.DataFrame(residuals))
        X[col] = residuals.resid + decomposed.trend + decomposed.seasonal
    return X



# stationarity check
def get_diff(data,target):
    diff=[]
    column= target
    
    trial=0
    values = data[column].values.tolist()
    cleanedList = [x for x in values if str(x) != 'nan']
    result = adfuller(cleanedList)
    # Extract the p-value from the test result
    p_value = result[1]
    for i in range(1, 4):
        #print(column,trial,p_value)
        if p_value < 0.05:
            diff.append(trial)
            break
        else:
            if trial==2:
                diff.append(0)
                break
            data[column] = data[column].diff(1)
            values = data[column].values.tolist()
            cleanedList = [x for x in values if str(x) != 'nan']
            result = adfuller(cleanedList)
            p_value = result[1]
            trial+=1
    return column,diff




# stationarity (applying difference)
def apply_diff(data,diff,column):
        
    #print(diff,columns)

    data[f"{column}_found"]=data[column]
    for j in range(diff[0]):
          data[f"{column}_found"] = data[f"{column}_found"].diff(1)
    return data




def detect_seasonality(df, target_column):

    # Ensure the DataFrame index is a valid date/time index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Decompose the time series to capture seasonality
    decomposition = sm.tsa.seasonal.seasonal_decompose(df[target_column], model='additive', extrapolate_trend='freq')

    # Test if the residuals exhibit any seasonality
    seasonal_component = decomposition.seasonal
    mean_seasonality = np.mean(seasonal_component)
    is_additive = np.allclose(seasonal_component, mean_seasonality, atol=1e-3)

    # Determine the seasonality type
    if is_additive:
        return "additive"
    else:
        return "multiplicative"


def preprocess_seasonality(df,target_column,seasonality_type="additive"):
    if seasonality_type == "multiplicative":
        df[target_column] = np.log(df[target_column])     
    df = df.reset_index() 
    return df   
    
    
def oheesimate(X):
    
    categorical_cols = [col for col in X.columns if X[col].dtype == 'string']
    encoder = None
    if categorical_cols:
        encoder = OneHotEncoder(variables=categorical_cols).fit(X)
    return categorical_cols,encoder

def oheeapply(X,categorical_cols,encoder):
    if categorical_cols:
        X = encoder.transform(X)
    return X
                

def lagger(X,max_lag,sig_fet):
    cols = [col for col in X.columns if 'found' in col]
    for col in cols:
        significant_lags = []
        m = 0
        pacf_values, conf_interval = pacf(X[col], nlags=max_lag, method='ols', alpha=0.05)

        result = np.zeros((len(pacf_values), 2))
        result[:, 0] = conf_interval[:len(pacf_values), 0] - pacf_values
        result[:, 1] = conf_interval[:len(pacf_values), 1] - pacf_values 

        new = result[1]
        for i in range(len(pacf_values)): 
            if i == 0:
                continue
            if pacf_values[i] < new[0] or pacf_values[i] > new[1]:
                significant_lags.append(i)
                sig_fet[col] = significant_lags

                m = 0
            else:
                m += 1
                if m == 3:
                    break
        
    return sig_fet

        
        
def lagger_apply(X,sig_fet):
    for key, value in sig_fet.items():
        for i in value:
            lag_column = X[key].shift(i)
            new_column_name = f"lag_{key}_{i}"
            X[new_column_name] = lag_column
    return X

        




def fourier_features(time, freq, order):
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, (order + 1)):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features)
    


def seasonality_transform(X,freq):

    time = X['ordered'].values

    fourier_series_features = fourier_features(time, freq,6)

    # Create DataFrame for Fourier series features
    fourier_df = fourier_series_features.set_index(X.index)

    # Concatenate the input DataFrame and Fourier series DataFrame
    combined_df = pd.concat([X, fourier_df], axis=1)
    return combined_df



def check_log(df,target_column,seasonality_type="additive"):
    if seasonality_type == "multiplicative":
        df[target_column] = np.exp(df[target_column])
        df['prediction'] = np.exp(df['prediction'])
    
    return df 

        
        
        
        
        
        
        
        
        