import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib


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
        self.forward = None
        self.backward = None
        standardizationTypes = ['differ', 'logdiffer', 'fractional', 'rollingstart', 'rollingend']
        if type not in standardizationTypes:
            raise ValueError('Type not implemented. Should be in ' + str(standardizationTypes))


    def sequenceConvert(self, input, output, back=[0], forward=[0]):
        '''

        :param input: input data
        :param output: output data
        :param back: which samples are fed as descriptors
        :param forward: which samples do we need to predict
        :return: differenced input/output in sequence. Output (samples, back, features), input (samples, prediction)
        '''
        if type(back) == int:
            back = [back]
        if type(forward) == int:
            forward = [forward]
        self.backward = np.array([int(x) for x in back])
        self.forward = np.array([int(x) for x in forward])

        max_back = max(self.backward)
        n_features = input.shape[1]
        n_samples = int(len(output) - max(self.backward) - max(self.forward))

        input_sequence = np.zeros((n_samples, len(self.backward), n_features))
        output_sequence = np.zeros((n_samples, len(self.forward)))

        if self.type == 'differ':
            n_samples -= 1
            input_sequence = np.zeros((n_samples, len(self.backward), n_features))
            output_sequence = np.zeros((n_samples, len(self.forward)))
            for i in range(n_samples):
                input_sequence[i] = input[i + self.backward + 1] - input[i + self.backward]
                output_sequence[i] = output[i + self.forward + 1 + max_back] - output [i + self.forward + max_back]

        elif self.type == 'logdiffer':
            if (np.min(input) < 0).any()  or (np.min(output) < 0).any():
                raise ValueError('Cannot take log of a negative value, ensure positive data')
            input = np.log(input)
            output = np.log(output)
            n_samples -= 1
            input_sequence = np.zeros((n_samples, len(self.backward), n_features))
            output_sequence = np.zeros((n_samples, len(self.forward)))
            for i in range(n_samples):
                input_sequence[i] = input[i + self.backward + 1] - input[i + self.backward]
                output_sequence[i] = output[i + 1 + max_back + self.forward] - output [i + max_back + self.forward]

        elif self.type == 'fractional':
            input_residual = 0
            output_residual = 0
            weights = self.getFractionalWeights(self.fraction)
            for k in range(self.lagCutOff):
                shifted = np.roll(input, k)
                shifted[:k] = 0
                input_residual += weights[k] * shifted
                shifted = np.roll(output, k)
                shifted[:k] = 0
                output_residual += weights[k] * shifted
            for i in range(n_samples):
                input_sequence[i] = input_residual[i + self.backward]
                output_sequence[i] = output_residual[i + self.forward + max_back]

        elif self.type == 'rollingstart':
            for i in range(n_samples):
                input_sequence[i] = input[i + self.backward] - input[i]
                output_sequence[i]  = output[i + max_back + self.forward] - output[i + max_back]

        elif self.type == 'rollingend':
            for i in range(n_samples):
                input_sequence[i] = input[i + self.backward] - input[i + max_back]
                output_sequence[i]  = output[i + max_back + self.forward] - output[i + max_back]

        return input_sequence, output_sequence


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
            weights = self.getFractionalWeights(self.fraction)
            for k in range(self.lagCutOff):
                shifted = np.roll(output, k)
                shifted[:k] = 0
                res += weights[k] * shifted
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


    def recreateOutput(self, output, initial):
        if self.type == 'differ':
            return np.vstack((initial, initial + np.cumsum(output, axis=0)))
        elif self.type == 'logdiffer':
            return np.vstack((initial, np.exp(np.log(initial) + np.cumsum(output, axis=0))))
        elif self.type == 'fractional':
            weights = self.getFractionalWeights(-self.fraction)
            res = 0
            for k in range(self.lagCutOff):
                shifted = np.roll(output, k, axis=0)
                shifted[:k] = 0
                res += weights[k] * shifted
            return np.hstack((initial, initial + res))
        elif self.type[:7] == 'rolling':
            if 1 not in self.forward:
                raise ValueError('To recreate rolling differenced signal, "1" needs to be within forward')
            indOne = np.where(self.forward == 1)[0][0]
            res = np.zeros_like(output)
            res[:, indOne] = initial + np.cumsum(output[:, indOne])
            original = res[:, indOne] - output[:, indOne]
            for i, ind in enumerate(self.forward):
                if ind == 1:
                    continue
                res[:, i] = original + output[:, i]
            return res


    def getFractionalWeights(self, fraction):
        '''
        Internal function what gets the weights of fractional differencing.
        :return: weights.
        '''
        w = [1]
        for k in range(1, self.lagCutOff):
            w.append(-w[-1] * (fraction - k + 1) / k)
        return np.array(w).reshape((-1))

class ExploratoryDataAnalysis(object):

    def __init__(self, data, differ=0, pretag=None, output=None, maxSamples=10000, seasonPeriods=[24 * 60, 7 * 24 * 60], lags=60):
        '''
        Doing all the fun EDA in an automized script :)
        :param data: Pandas Dataframe
        :param output: Pandas series of the output
        :param seasonPeriods: List of periods to check for seasonality
        :param lags: Lags for (P)ACF and
        '''
        # Register data
        matplotlib.use('Agg')
        matplotlib.rcParams['agg.path.chunksize'] = 1000000
        self.differ = differ
        self.tag = pretag
        self.maxSamples = maxSamples
        self.data = data
        self.output = output
        self.seasonPeriods = seasonPeriods
        self.lags = lags

        # Create dirs
        self.createDirs()

        # Run all functions
        print('Generating Timeplots')
        self.timeplots()
        print('Generating Boxplots')
        self.boxplots()
        # self.seasonality()
        print('Generating Colinearity Plots')
        self.colinearity()
        print('Generating Diff Var Plot')
        self.differencing()
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
        dirs = ['EDA', 'EDA/Boxplots', 'EDA/Seasonality', 'EDA/Colinearity', 'EDA/Lags', 'EDA/Correlation',
                'EDA/Correlation/ACF', 'EDA/Correlation/PACF', 'EDA/Correlation/Cross', 'EDA/NonLinear Correlation', 'EDA/Timeplots']
        for period in self.seasonPeriods:
            dirs.append('EDA/Seasonality/' + str(period))
        for dir in dirs:
            try:
                os.mkdir(dir)
                if dir == 'EDA/Correlation':
                    file = open('EDA/Correlation/Readme.txt', 'w')
                    edit = file.write(
                        'Correlation Interpretation\n\nIf the series has positive autocorrelations for a high number of lags,\nit probably needs a higher order of differencing. If the lag-1 autocorrelation\nis zero or negative, or the autocorrelations are small and patternless, then \nthe series does not need a higher order of differencing. If the lag-1 autocorrleation\nis below -0.5, the series is likely to be overdifferenced. \nOptimum level of differencing is often where the variance is lowest. ')
                    file.close()
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
        # Undersample
        data = self.data.iloc[np.linspace(0, len(self.data) - 1, self.maxSamples).astype('int')]
        # Plot
        for key in data.keys():
            fig = plt.figure(figsize=[24, 16])
            plt.plot(data.index, data[key])
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
        threshold = 0.95
        fig = plt.figure(figsize=[24, 16])
        sns.heatmap(abs(self.data.corr()) < threshold, annot=False, cmap='Greys')
        fig.savefig('EDA/Colinearity/' + self.tag + 'All_Features.png', format='png', dpi=300)
        # Minimum representation
        corr_mat = self.data.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        minimal_rep = self.data.drop(self.data[col_drop], axis=1)
        fig = plt.figure(figsize=[24, 16])
        sns.heatmap(abs(minimal_rep.corr()) < threshold, annot=False, cmap='Greys')
        fig.savefig('EDA/Colinearity/' + self.tag + 'Minimum_Representation.png', format='png', dpi=300)

    def differencing(self):
        max_lags = 4
        varVec = np.zeros((max_lags, len(self.data.keys())))
        diffData = self.data / np.sqrt(self.data.var())
        for i in range(max_lags):
            varVec[i, :] = diffData.var()
            diffData = diffData.diff(1)[1:]
        fig = plt.figure(figsize=[24, 16])
        plt.yscale('log')
        plt.plot(varVec)
        plt.title('Variance for different lags')
        plt.xlabel('Lag')
        plt.ylabel('Average variance')
        fig.savefig('EDA/Lags/'  + self.tag + 'Variance.png', format='png', dpi=300)

    def completeAutoCorr(self):
        for i in range(self.differ):
            self.data = self.data.diff(1)[1:]
        for key in self.data.keys():
            fig = plot_acf(self.data[key], fft=True)
            plt.title(key)
            fig.savefig('EDA/Correlation/ACF/' + self.tag + key + '_differ_ ' + str(self.differ) + '.png', format='png', dpi=300)
            plt.close()

    def partialAutoCorr(self):
        for key in self.data.keys():
            fig = plot_pacf(self.data[key], fft=True)
            fig.savefig('EDA/Correlation/PACF/' + self.tag + key + '_differ_ ' + str(self.differ) + '.png', format='png', dpi=300)
            plt.title(key)
            plt.close()

    def crossCorr(self):
        for key in self.data.keys():
            fig = plt.figure(figsiz=[24, 16])
            plt.xcorr(self.data[key], self.output, maxlags=self.lags)
            plt.title(key)
            fig.savefig('EDA/Correlation/Cross/' + self.tag + key + '_differ_ ' + str(self.differ) + '.png', format='png', dpi=300)
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

class Modelling(object):

    def __init__(self):
        print('To be implemented!')

class Metrics(object):

    def r2score(ytrue, ypred):
        res = sum((ytrue - ypred) ** 2)
        tot = sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - res / tot

    def mae(ytrue, ypred):
        return np.mean(abs(ytrue - ypred))

    def mse(ytrue, ypred):
        return np.mean((ytrue - ypred) ** 2)
