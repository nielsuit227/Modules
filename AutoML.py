import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use('Agg')


class Preprocessing(object):

    def __init__(self,
                 inputPath,
                 outputPath,
                 indexCol=None,
                 datesCols=None,
                 stringCols=None,
                 missingValues='remove',
                 outlierRemoval='none',
                 zScoreThreshold=4,
                 ):
        '''
        Preprocessing Class. Deals with Outliers, Missing Values, duplicate rows, data types (floats, categorical and dates),
        NaN, Infs.

        :param inputPath: Path to file or directory of files to be processed
        :param outputPath: Path to write processed files
        :param indexCol: Whether or not to take an index column from the input files.
        :param datesCols: Date columns, all parsed through pd.to_datetime()
        :param stringCols: String Columns. all parsed as categorical (output cat codes)
        :param missingValues: How to deal with missing values ('remove', 'interpolate' or 'mean')
        :param outlierRemoval: How to deal with outliers ('boxplot', 'zscore' or 'none')
        :param zScoreThreshold: If outlierRemoval='zscore', the threshold is adaptable, default=4.
        '''
        missingValuesImplemented = ['remove', 'interpolate', 'mean']
        outlierRemovalImplemented = ['boxplot', 'zscore', 'none']
        if outlierRemoval not in outlierRemovalImplemented:
            raise ValueError("Outlier Removal Algo not implemented. Should be in " + str(outlierRemovalImplemented))
        if missingValues not in missingValuesImplemented:
            raise ValueError("Missing Values Algo not implemented. Should be in " + str(missingValuesImplemented))
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.indexCol = indexCol
        self.datesCols = datesCols
        self.stringCols = stringCols
        self.missingValues = missingValues
        self.outlierRemoval = outlierRemoval
        self.zScoreThreshold = zScoreThreshold
        try:
            files = os.listdir(inputPath)
            for file in files:
                print(file)
                self._clean(file)
        except:
            self._clean('')


    def _clean(self, path):
        '''
        Internal function
        :param path: path to file.
        :return:
        '''
        # Read data
        data = pd.read_csv(self.inputPath + path, index_col=self.indexCol, parse_dates=self.datesCols)

        # Clean keys
        newKeys = {}
        for key in data.keys():
            newKeys[key] = key.lower().replace('.', '_').replace('/', '_')
        data = data.rename(columns=newKeys)

        # Convert dtypes
        if self.datesCols:
            keys = [x for x in data.keys() if x not in self.datesCols]
        if self.stringCols:
            keys = [x for x in keys if x not in self.stringCols]
            for key in self.stringCols:
                data[key] = data[key].astype('category').cat.codes
        for key in keys:
            data[key] = pd.to_numeric(data[key], errors='coerce').astype('float32')

        # Duplicates
        data = data.drop_duplicates()

        # Remove Anomalies
        if self.outlierRemoval == 'boxplot':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            for key in Q1.keys():
                data.loc[data[key] < Q1[key] - 1.5 * (Q3[key] - Q1[key]), key] = np.nan
                data.loc[data[key] > Q3[key] + 1.5 * (Q3[key] - Q1[key]), key]
                # data[key][data[key] < Q1[key] - 1.5 * (Q3[key] - Q1[key])] = np.nan
                # data[key][data[key] > Q3[key] + 1.5 * (Q3[key] - Q1[key])] = np.nan
        elif self.outlierRemoval == 'zscore':
            Zscore = (data - data.mean(skipna=True, numeric_only=True)) \
                     / np.sqrt(data.var(skipna=True, numeric_only=True))
            data[Zscore > self.zScoreThreshold] = np.nan

        # Missing values etc.
        data = data.replace([np.inf, -np.inf], np.nan)
        if self.missingValues == 'remove':
            data = data[data.isna().sum(axis=1) == 0]
        elif self.missingValues == 'interpolate':
            data = data.interpolate()
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif self.missingValues == 'mean':
            data = data.fillna(data.mean())

        # Save
        print('Saving to: ', self.outputPath + path)
        data.to_csv(self.outputPath + path, index=self.indexCol is not None)

class Normalize(object):

    def __init__(self,
                 type='normal'):
        '''
        Normalization class.
        :param type: Either minmax, scales to 0-1, or normal, scaling to a normal distribution.
        '''
        self.type = type
        self.mean = None
        self.var = None
        self.max = None
        self.min = None
        normalizationTypes = ['normal', 'minmax']
        if type not in normalizationTypes:
            raise ValueError('Type not implemented. Should be in ' + str(normalizationTypes))

    def convert(self, data):
        '''
        Actual Conversion
        :param data: Pandas DataFrame.
        :return:
        '''
        self.mean = data.mean()
        self.var = data.var()
        self.min = data.min()
        self.max = data.max()
        if self.type == 'normal':
            data -= self.mean
            data /= np.sqrt(self.var)
        elif self.type == 'minmax':
            data -= self.min
            data /= self.max - self.min
        return data

class Stationarize(object):

    def __init__(self, type='differ', fraction=0.5, lagCutOff=5, sequence=1):
        '''
        Initialize the stationarizing function.
        :param type: Type of stationarization, options: 'differ', 'logdiffer', 'fractional', 'rollingstart', 'rollingend'
        :param fraction: if type='fractional', the fraction of fractional differencing
        :param lagCutOff: type='fractional' results in infinite decaying weights, which are cutoff by lagCutOff.
        :param sequence: if 'rolling', a rolling differences sequence is returned of length seq.
        '''
        self.type = type
        self.fraction = fraction
        self.lagCutOff = lagCutOff
        self.sequence = sequence
        standardizationTypes = ['differ', 'logdiffer', 'fractional', 'rollingstart', 'rollingend']
        if type not in standardizationTypes:
            raise ValueError('Type not implemented. Should be in ' + str(standardizationTypes))


    def convert(self, data):
        '''
        Function that does the actual conversion.
        :param data: Pandas Dataframe.
        :return:
        '''
        if self.type == 'differ':
            return data.diff()[1:]
        elif self.type == 'logdiffer':
            return np.log(data).diff()[1:]
        elif self.type == 'fractional':
            res = 0
            weights = self.getFractionalWeights()
            for k in range(self.lagCutOff):
                res += weights[k] * data.shift(k).fillna(0)
            return res[self.lagCutOff:]
        elif self.type == 'rollingstart':
            nSamples = len(data) - self.sequence
            seqX = np.zeros((nSamples, self.sequence, len(data.keys())))
            for i in range(nSamples):
                seqX[i] = data.iloc[i:i + self.sequence].to_numpy() - data.iloc[i]
            return seqX
        elif self.type == 'rollingend':
            nSamples = len(data) - self.sequence
            seqX = np.zeros((nSamples, self.sequence, len(data.keys())))
            for i in range(nSamples):
                seqX[i] = data.iloc[i:i + self.sequence].to_numpy() - data.iloc[i + self.sequence]
            return seqX


    def getFractionalWeights(self):
        '''
        Internal function what gets the weights of fractional differencing.
        :return: weights.
        '''
        w = [1]
        for k in range(1, self.lagCutOff):
            w.append(-w[-1] * (self.fractional - k + 1) / k)
        return np.array(w).reshape((-1))

class ExploratoryDataAnalysis(object):

    def __init__(self, data, pretag=None, output=None, seasonPeriods=[24 * 60, 7 * 24 * 60], lags=60):
        '''
        Doing all the fun EDA in an automized script :)
        :param data: Pandas Dataframe
        :param output: Pandas series of the output
        :param seasonPeriods: List of periods to check for seasonality
        :param lags: Lags for (P)ACF and
        '''
        # Register data
        self.tag = pretag
        self.data = data
        self.output = output
        self.seasonPeriods = seasonPeriods
        self.lags = lags

        # Create dirs
        self.createDirs()

        # Run all functions
        print('Generating Boxplots')
        self.boxplots()
        print('Generating Timeplots')
        self.timeplots()
        # self.seasonality()
        print('Generating Colinearity Plots')
        self.colinearity()
        print('Generating ACF Plots')
        self.completeAutoCorr()
        print('Generating PACF Plots')
        self.partialAutoCorr()
        if self.output is not None:
            print('Generating CCF Plots')
            self.crossCorr()
            print('Generating Feature Ranking Plot')
            self.featureRanking()

    def createDirs(self):
        dirs = ['EDA', 'EDA/Boxplots', 'EDA/seasonality', 'EDA/Colinearity', 'EDA/Lags', 'EDA/Correlation/ACF',
                    'EDA/Correlation/Cross', 'EDA/NonLinear Correlation', 'EDA/Timeplots']
        for period in self.seasonPeriods:
            dirs.append('EDA/Seasonality' + str(period))
        for dir in dirs:
            try:
                os.mkdir(dir)
            except:
                continue

    def boxplots(self):
        for key in self.data.keys():
            fig = plt.figure(figsize=[24, 16])
            plt.boxplot(self.data[key])
            plt.title(key)
            fig.savefig('EDA/Boxplots/' + self.tag + key + '.png', format='png', dpi=300)
            plt.close()

    def timeplots(self):
        for key in self.data.keys():
            fig = plt.figure(figsize=[24, 16])
            plt.plot(self.data[key])
            plt.title(key)
            fig.savefig('EDA/Timeplots/' + self.tag + key + '.png', format='png', dpi=300)
            plt.close()

    def seasonality(self):
        for period in self.seasonPeriods:
            for key in self.data.keys():
                seasonality = STL(self.data[key], period=period).fit()
                fig = plt.figure(figsize=[24, 16])
                plt.plot(range(len(self.data)), self.data[key])
                plt.plot(range(len(self.data)), seasonality)
                plt.title(key + ', period=' + str(period))
                fig.savefig('EDA/Seasonality/' + self.tag + str(period)+'/'+key + '.png', format='png', dpi=300)
                plt.close()

    def colinearity(self):
        fig = plt.figure(figsize=[24, 16])
        sns.heatmap(self.data.corr(), annot=False, cmap='bwr')
        fig.savefig('EDA/Colinearity/' + self.tag + 'All_Features.png', format='png', dpi=300)
        # Minimum representation
        corr_mat = self.data.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        minimal_rep = self.data.drop(self.data[col_drop], axis=1)
        fig = plt.figure(figsize=[24, 16])
        sns.heatmap(minimal_rep.corr(), annot=False, cmap='bwr')
        fig.savefig('EDA/Colinearity/' + self.tag + 'Minimum_Representation.png', format='png', dpi=300)

    def differencing(self):
        max_lags = 4
        varVec = np.zeros(max_lags)
        diffData = self.data
        for i in range(max_lags):
            varVec[i] = diffData.var().mean()
            diffData = diffData.diff(1)[1:]
        fig = plt.figure(figsize=[24, 16])
        plt.yscale('log')
        plt.plot(varVec)
        plt.title('Variance for different lags')
        plt.xlabel('Lag')
        plt.ylable('Average variance')
        fig.savefig('EDA/Lags/'  + self.tag + 'Variance.png', format='png', dpi=300)

    def completeAutoCorr(self):
        file = open('EDA/Correlation/Readme.txt', 'wb')
        file.write('Correlation Interpretation\n\n'
                   'If the series has positive autocorrelations for a high number of lags,\n'
                   'it probably needs a higher order of differencing. If the lag-1 autocorrelation\n'
                   'is zero or negative, or the autocorrelations are small and patternless, then \n'
                   'the series does not need a higher order of differencing. If the lag-1 autocorrleation\n'
                   'is below -0.5, the series is likely to be overdifferenced. \n'
                   'Optimum level of differencing is often where the variance is lowest. ')
        file.close()
        for key in self.data.keys():
            fig = plt.figure(figsize=[24, 16])
            plot_acf(self.data[keys])
            plt.title(key)
            fig.savefig('EDA/Correlation/Auto/' + self.tag + key + '.png', format='png', dpi=300)
            plt.close()

    def partialAutoCorr(self):
        for key in self.data.keys():
            fig = plt.figure(figsize=[24, 16])
            plot_pacf(self.data[keys])
            fig.savefig('EDA/Correlation/PACF/' + self.tag + key + '.png', format='png', dpi=300)
            plt.title(key)
            plt.close()

    def crossCorr(self):
        for key in self.data.keys():
            fig = plt.figure(figsiz=[24, 16])
            plt.xcorr(self.data[key], self.output, maxlags=self.lags)
            plt.title(key)
            fig.savefig('EDA/Correlation/Cross/' + self.tag + key + '.png', format='png', dpi=300)
            plt.close()

    def featureRanking(self):
        if set(self.output) == {1, -1}:
            model = RandomForestClassifier().fit(self.data, self.output)
        else:
            model = RandomForestRegressor().fit(self.data, self.output)
        fig = plt.figure(figsize=[24, 16])
        ind = np.argsort(model.feature_importances_, reversed=True)
        plt.bar(self.data.keys()[ind], model.feature_importances_[ind])
        plt.xticks(rotation=60)
        fig.savefig('EDA/Nonlinear Correlation/' + self.tag + 'RF.png', format='png', dpi=300)
        plt.close()


