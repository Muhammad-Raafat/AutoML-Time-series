# import subprocess
# def install_library(library_name):
#     subprocess.check_call(['pip', 'install', library_name])
    
# # Usage
# install_library('feature_engine')

from statsmodels.tsa.seasonal import STL
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import pacf




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
def get_diff(data):
    diff=[]
    columns=data.select_dtypes(include="number").columns.tolist()
    for column in columns:
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
    return columns,diff




# stationarity (applying difference)
def apply_diff(data,diff,columns):
        
    #print(diff,columns)
    for i in range(len(columns)):
        data[f"{columns[i]}_found"]=data[columns[i]]
        for j in range(diff[i]):
            data[f"{columns[i]}_found"] = data[f"{columns[i]}_found"].diff(1)
    return data





def fourier_series_features_creator(time_series , time , n,num_terms,significant_periods):

    # Generate Fourier series features
    fourier_series_features = np.zeros((n, num_terms * len(significant_periods) * 2))
    for i, period in enumerate(significant_periods):
        for j in range(num_terms):
            coefficient_sin = np.sin(2 * np.pi * (j + 1) * time / period)
            coefficient_cos = np.cos(2 * np.pi * (j + 1) * time / period)
            fourier_series_features[:, i * num_terms + j] = coefficient_sin
            fourier_series_features[:, i * num_terms + j + num_terms] = coefficient_cos
    return fourier_series_features






def get_stationarity(X,target,power_threshold,length_threshold,):
    time_series = X[target]
    time = X[target].rank(method='dense').astype(int)

    # Linear detrend
    detrended_series = time_series.values.astype("float") - np.polyval(np.polyfit(time, time_series.values.astype("float"), 1), time)

    # Compute the periodogram with boxcar window and spectrum scaling
    n = len(detrended_series)
    power_spectrum = np.abs(fft(detrended_series * np.hanning(n)))**2 / n

    # Apply power threshold and length threshold
    significant_indices = np.where(power_spectrum > self.power_threshold)[0][1:]
    significant_periods = 1 / fftfreq(n)[significant_indices]
    self.significant_periods = significant_periods[np.where(significant_periods > self.length_threshold)[0]]

    self.fourier_series_features_creator(time_series, time, n)

    # Create DataFrame for Fourier series features
    self.feature_columns = [f'Feature_{i+1}' for i in range(self.fourier_series_features.shape[1])]
    self.fourier_df = pd.DataFrame(self.fourier_series_features, columns=self.feature_columns)


    
    
    
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

        

def fourier_series_features_creator(time, num_terms, significant_periods , n):
    # Generate Fourier series features
    fourier_series_features = np.zeros((n, num_terms * len(significant_periods) * 2))
    for i, period in enumerate(significant_periods):
        for j in range(num_terms):
            coefficient_sin = np.sin(2 * np.pi * (j + 1) * time / period)
            coefficient_cos = np.cos(2 * np.pi * (j + 1) * time / period)
            fourier_series_features[:, i * num_terms + j] = coefficient_sin
            fourier_series_features[:, i * num_terms + j + num_terms] = coefficient_cos
    return fourier_series_features




def seasonality_fit(X , target,power_threshold, length_threshold, num_terms):
    # Assuming you have an existing DataFrame named X
    time_series = X[target]
    time = X[target].rank(method='dense').astype(int)

    # Linear detrend
    detrended_series = time_series.values.astype("float") - np.polyval(
                        np.polyfit(time, time_series.values.astype("float"), 1), time)

    # Compute the periodogram with boxcar window and spectrum scaling
    n = len(detrended_series)
    power_spectrum = np.abs(fft(detrended_series * np.hanning(n))) ** 2 / n

    # Apply power threshold and length threshold
    significant_indices = np.where(power_spectrum > power_threshold)[0][1:]
    significant_periods = 1 / fftfreq(n)[significant_indices]
    significant_periods = significant_periods[np.where(significant_periods > length_threshold)[0]]

    fourier_series_features = fourier_series_features_creator(time, num_terms, significant_periods, n)

    # Create DataFrame for Fourier series features
    feature_columns = [f'Feature_{i + 1}' for i in range(fourier_series_features.shape[1])]

    return significant_periods, feature_columns







def seasonality_transform(X, target,significant_periods, feature_columns,num_terms ):
    # Assuming you have an existing DataFrame named X
    time_series = X[target]
    time = X[target].rank(method='dense').astype(int)

    fourier_series_features = fourier_series_features_creator(time, num_terms, significant_periods, len(X))

    # Create DataFrame for Fourier series features
    fourier_df = pd.DataFrame(fourier_series_features, columns=feature_columns ).set_index(X.index)

    # Concatenate the input DataFrame and Fourier series DataFrame
    combined_df = pd.concat([X, fourier_df], axis=1)
    return combined_df





        
        
        
        
        
        
        
        
        