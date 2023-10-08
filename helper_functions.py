import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# create a class for cleaning data
class cleaning_functions:
    def __init__(self, df):
        self.df = df
    
    # delete columns with atleast 80% null values
    def delete_80_null(self):
        """
        This function will delete columns with at least 80% null values
        """
        # Calculate the threshold for null values
        threshold = 0.8 * len(self.df)
        # Get columns where null values are >= threshold
        null_columns = list(self.df.columns[self.df.isnull().sum() >= threshold])
        # Get the columns that are not in the null_columns
        not_null_columns = self.df.columns[~self.df.columns.isin(null_columns)]
        self.df = self.df[not_null_columns]
        # Return the dataframe
        return self.df


    # check for duplicate rows
    def check_duplicates(self):
        """
        This function will check for duplicate rows in the dataframe
        """
        # get the total number of duplicate rows
        total = self.df.duplicated().sum()
        # print the total number of duplicate rows
        total_message = f'Total number of duplicate rows: {total}'
        # return the total number of duplicate rows
        return total_message

    # check for missing values
    def check_missing_values(self):
        """
        This function will check for missing values in the dataframe
        """
        # get the total number of missing values
        total = self.df.isnull().sum().sort_values(ascending=False)
        # get the percentage of missing values
        percent = round((self.df.isnull().sum()/self.df.isnull().count()*100),2).sort_values(ascending=False)
        # create a dataframe of the missing values
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        # return the dataframe
        return missing_data.T

    # check data types
    def check_datatypes(self):
        """
        This function will check the data types of the columns in the dataframe
        """
        # get the data types of the columns
        data_types = pd.DataFrame(self.df.dtypes).T
        # return the data types
        return data_types

    # fill missing values
    def fill_missing_values(self, column, value):
        """
        This function will fill the missing values in the column with the specified value
        """
        # fill the missing values in the column with the specified value
        self.df[column] = self.df[column].fillna(value)
        # return the dataframe
        return self.df


    # remove percent sign from the column
    def remove_percent_sign(self, column):
        """
        This function will remove the percent sign from the column
        """
        # remove the percent sign from the column
        self.df[column] = self.df[column].str.replace('%', '')
        # return the dataframe
        return self.df

    # remove other delimeter from the column
    def remove_delimeter(self, column, delimeter=','):
        """
        This function will remove the delimeter from the column
        """
        # remove the delimeter from the column
        self.df[column] = self.df[column].str.replace(delimeter, '')
        # return the dataframe
        return self.df


    # convert column data type
    def convert_datatype(self, column, to_type):
        """
        This function will convert the column to the specified type
        """
        if to_type == 'datetime':
            self.df[column] = pd.to_datetime(self.df[column])
        elif to_type == 'numeric':
            self.df[column] = pd.to_numeric(self.df[column])
        elif to_type == 'category':
            self.df[column] = self.df[column].astype('category')
        else:
            print("Invalid type specified. Please specify 'datetime', 'numeric', or 'category'.")
        return self.df


# create a class for feature engineering
class feature_engineering_functions:
    def __init__(self, df):
        self.df = df

    # create time features
    def create_time_features(self, column):
        """
        This function will create time features from the specified column
        """
        # create time features
        self.df['year'] = self.df[column].dt.year
        self.df['month'] = self.df[column].dt.strftime('%B')
        self.df['day_of_week'] = self.df[column].dt.strftime('%A')
        self.df['hour'] = self.df[column].dt.hour
        self.df['is_weekend'] = self.df[column].dt.weekday.apply(lambda x: 1 if x > 4 else 0)
        # return the dataframe
        return self.df
    
    # create has_hashtag feature
    def create_hashtag_feature(self, column):
        """
        This function will create a feature that will check if a post has a hashtag
        """
        # create the feature
        self.df['has_hashtag'] = self.df[column].apply(lambda x: True if '#' in x else False)
        # return the dataframe
        return self.df

    # create number of hashtags feature
    def create_num_hashtags_feature(self, column):
        """
        This function will create a feature that will count the number of hashtags in a post
        """
        # create the feature
        self.df['num_hashtags'] = self.df[column].apply(lambda x: len([c for c in x if c == '#']))
        # return the dataframe
        return self.df


# create a class for time series analysis
class time_series_functions:
    def __init__(self, df):
        self.df = df

    # check for stationarity
    def check_stationarity(self, column):
        """
        This function will check for stationarity in the time series data
        """
        # import the adfuller module
        from statsmodels.tsa.stattools import adfuller
        # get the p-value
        p_value = adfuller(self.df[column])[1]
        # return the p-value
        return p_value

    # plot the time series data
    def plot_time_series(self, column):
        """
        This function will plot the time series data
        """
        # plot the time series data
        plt.plot(self.df[column])
        # show the plot
        plt.show()

    # plot the autocorrelation plot
    def plot_autocorrelation(self, column):
        """
        This function will plot the autocorrelation plot
        """
        # import the statsmodels module
        from statsmodels.graphics.tsaplots import plot_acf
        # plot the autocorrelation plot
        plot_acf(self.df[column])
        # show the plot
        plt.show()

    # plot the partial autocorrelation plot
    def plot_partial_autocorrelation(self, column):
        """
        This function will plot the partial autocorrelation plot
        """
        # import the statsmodels module
        from statsmodels.graphics.tsaplots import plot_pacf
        # plot the partial autocorrelation plot
        plot_pacf(self.df[column])
        # show the plot
        plt.show()

    # plot the seasonal decomposition plot
    def plot_seasonal_decomposition(self, column):
        """
        This function will plot the seasonal decomposition plot
        """
        # import the statsmodels module
        from statsmodels.tsa.seasonal import seasonal_decompose
        # get the seasonal decomposition
        seasonal_decompose(self.df[column]).plot()
        # show the plot
        plt.show()

    # plot the acf and pacf plots
    def plot_acf_pacf(self, column):
        """
        This function will plot the acf and pacf plots
        """
        # import the statsmodels module
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        # plot the acf and pacf plots
        fig, ax = plt.subplots(1,2,figsize=(16,4))
        plot_acf(self.df[column], ax=ax[0])
        plot_pacf(self.df[column], ax=ax[1])
        # show the plot
        plt.show()
