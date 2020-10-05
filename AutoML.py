import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Preprocessing(object):

    def __init__(self,
                 inputPath,
                 outputPath,
                 indexCol=None,
                 datesCols=None,
                 stringCols=None,
                 missingValues='remove',
                 outlierRemoval='boxplot',
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
            self._clean(inputPath)


    def _clean(self, path):
        '''
        Internal function
        :param path: path to file.
        :return:
        '''
        # Read data
        data = pd.read_csv(self.inputPath + path, index_col=self.indexCol, parse_dates=self.datesCols)

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
        elif self.missingValues == 'mean':
            data = data.fillna(data.mean())

        # Save
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

    def __init__(self, input, output):
        '''
        Doing all the fun EDA in an automized script :)
        :param input:
        :param output:
        '''
        # Create Directories
        os.mkdir('EDA/Boxplots')
        os.mkdir('EDA/Seasonality/Daily')
        os.mkdir('EDA/Seasonality/Weekly')
        os.mkdir('EDA/Seasonality/Montly')
        os.mkdir('EDA/Seasonality/Annually')
        os.mkdir('EDA/Colinearity')
        os.mkdir('EDA/Correlation/Auto')
        os.mkdir('EDA/Correlation/Auto - Diff')
        os.mkdir('EDA/Correlation/Cross')
        os.mkdir('EDA/Correlation/Cross - Diff')
        os.mkdir('EDA/NonLinear Correlation')















