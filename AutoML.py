import os, time, itertools, re
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# todo multiprocessing for EDA RF, GridSearch
# todo seq with pandas
# todo flat=bool for sequence (needed for lightGBM)
# todo implement PPS in EDA
# todo implement SHAP in EDA
# todo EDA to pandas
# todo implement autoencoder for Feature Extraction

# todo Determine Lag based on Auto/Cross Correlations

class Pipeline(object):
    def __init__(self, target,
                 max_lags=15,
                 info_threshold=0.9,
                 max_diff=1,
                 shift=[0]):
        print('\n\n*** Start Amplo PM Model Builder ***\n\n')
        assert type(target) == str;
        assert type(shift) == list;
        assert type(info_threshold) == float;
        assert type(max_diff) == int;
        assert max_lags < 50;
        assert info_threshold > 0 and info_threshold < 1;
        assert max_diff < 5;
        assert max(shift) >= 0 and min(shift) < 50;
        self.target = re.sub('[^a-zA-Z0-9 \n\.]', '_', target.lower())
        self.max_lags = max_lags
        self.info_threshold = info_threshold
        self.max_diff = max_diff
        self.shift = shift
        self.catKeys = []
        self.dateKeys = []
        self.norm = Normalize(type='minmax')
        self.prep = None
        self.seq = None
        self.diff = 'none'
        self.diffOrder = 0
        self.classification = False
        self.regression = True

    def fit(self, data):
        # Clean
        self.prep = Preprocessing(missingValues='interpolate')
        data = self.prep.clean(data)

        # Check whether is classification
        if len(set(data[self.target])) == 2:
            self.classification = True
            self.regression = False

        # Normalize
        data = self.norm.convert(data)

        # Stationarity Check
        varVec = np.zeros((self.max_diff + 1, len(data.keys())))
        diffData = data.copy(deep=True)
        for i in range(self.max_diff + 1):
            varVec[i, :] = diffData.std()
            diffData = diffData.diff(1)[1:]
        self.diffOrder = np.argmin(np.sum(varVec, axis=1))
        if self.diffOrder == 0:
            self.diff = 'none'
        else:
            self.diff = 'diff'
        print('[autoML] Optimal Differencing order: %i' % self.diffOrder)

        # Differencing, shifting, lagging
        n_features = len(data.keys())
        self.seq = Sequence(back=self.max_lags, forward=self.shift)
        if self.shift != 0 and self.diffOrder != 0:
            self.seq.diff = 'diff'
            data, output = self.seq.convert_pandas(data, data[[self.target]])
            data[self.target] = output
        elif self.shift != 0:
            data[self.target] = data[self.target].shift(-self.shift)
            data = data.iloc[:-self.shift]
        elif self.diffOrder != 0:
            for i in range(self.diffOrder):
                data = data.diff(1)[1:]
        print('[autoML] Added %i lags' % (len(data.keys()) - n_features))

        # Cross Correlation
        # cors = np.zeros((self.max_lags, len(input.keys())))
        # for i, key in enumerate(input.keys()):
        #     for j, lag in enumerate(range(self.max_lags)):
        #         cors[j, i] = np.correlate(input[key][:-1-j], output.iloc[j+1:])
        #     cors[j, :] /= max(cors[j, :])

        # Remove Colinearity
        print('[autoML] Removing Co-Linearity')
        nk = len(data.keys())
        norm = (data - data.mean(skipna=True, numeric_only=True)).to_numpy()
        ss = np.sqrt(np.sum(norm ** 2, axis=0))
        corr_mat = np.eye(nk)
        for i in range(nk):
            for j in range(nk):
                if i == j:
                    continue
                if corr_mat[i, j] == 0:
                    c = abs(np.sum(norm[:, i] * norm[:, j]) / (ss[i] * ss[j]))
                    corr_mat[i, j] = c
                    corr_mat[j, i] = c
        # Crashes here 'Pipeline' object has no attribute 'data'
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if self.target in col_drop:
            col_drop.remove(self.target)
        minimal_rep = self.data.drop(self.data[col_drop], axis=1)
        print('[autoML] Dropped %i Co-Linear variables.' % len(col_drop))

        # Keep based on PPScore
        pp_score = pps.score(lagged, self.target)
        col_keep = pp_score[pp_score['ppscore'] >= 0.9]['x'].tolist()
        input = lagged[col_keep]

        # Initial Modelling
        init = Modelling(input,)


        # Hyperparameter Optimization


class Preprocessing(object):

    def __init__(self,
                 inputPath=None,
                 outputPath=None,
                 indexCol=None,
                 missingValues='interpolate',
                 outlierRemoval='none',
                 zScoreThreshold=4,
                 ):
        '''
        Preprocessing Class. Deals with Outliers, Missing Values, duplicate rows, data types (floats, categorical and dates),
        NaN, Infs.

        :param input: (optional) Input data, Pandas DataFrame
        :param inputPath: (optional) Path to file or directory of files to be processed
        :param outputPath: (optional) Path to write processed files
        :param indexCol: Whether or not to take an index column from the input files.
        :param datesCols: Date columns, all parsed through pd.to_datetime()
        :param stringCols: String Columns. all parsed as categorical (output cat codes)
        :param missingValues: How to deal with missing values ('remove', 'interpolate' or 'mean')
        :param outlierRemoval: How to deal with outliers ('boxplot', 'zscore' or 'none')
        :param zScoreThreshold: If outlierRemoval='zscore', the threshold is adaptable, default=4.
        '''
        ### Data
        self.inputPath = inputPath
        self.outputPath = outputPath

        ### Algorithms
        missingValuesImplemented = ['remove', 'interpolate', 'mean']
        outlierRemovalImplemented = ['boxplot', 'zscore', 'none']
        if outlierRemoval not in outlierRemovalImplemented:
            raise ValueError("Outlier Removal Algo not implemented. Should be in " + str(outlierRemovalImplemented))
        if missingValues not in missingValuesImplemented:
            raise ValueError("Missing Values Algo not implemented. Should be in " + str(missingValuesImplemented))
        self.missingValues = missingValues
        self.outlierRemoval = outlierRemoval
        self.zScoreThreshold = zScoreThreshold

        ### Columns
        self.numCols = []
        self.catCols = []
        self.dateCols = []
        if indexCol:
            self.indexCol = [re.sub('[^a-zA-Z0-9 \n\.]', '_', x.lower()) for x in indexCols]
        else:
            self.indexCol = None

        ### If inputPath:
        if self.inputPath:
            files = os.listdir(inputPath)
            for file in files:
                data = pd.read_csv(self.inputPath + path,
                                   index_col=self.indexCol,
                                   parse_dates=self.datesCols)
                self.clean(data)


    def clean(self, data):
        print('[preprocessing] Data Cleaning Started')
        # Clean column names
        newKeys = {}
        for key in data.keys():
            newKeys[key] = re.sub('[^a-zA-Z0-9 \n\.]', '_', key.lower())
        data = data.rename(columns=newKeys)

        # Identify data types & convert
        missingThreshold = 0.5
        integerThreshold = 0.01
        for key in data.keys():
            # Check for numeric first
            numeric = pd.to_numeric(data[key], errors='coerce')
            if numeric.isna().sum() < len(data) * missingThreshold:
                if numeric.max() > 943920000:
                    dates = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True)
                    if np.logical_and(dates > pd.to_datetime('2000-01-01'),
                        dates < datetime.now()).sum() > len(data) * missingThreshold:
                        self.dateCols.append(key)
                        data[key] = pd.to_datetime(data[key])
                        continue
                self.numCols.append(key)
                data[key] = numeric
            else:
                dates = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True)
                if np.logical_and(dates > pd.to_datetime('2000-01-01'),
                    dates < datetime.now()).sum() > len(data) * missingThreshold:
                    self.dateCols.append(key)
                    data[key] = pd.to_datetime(data[key])
                    continue
                try:
                    dummies = pd.get_dummies(data[key])
                    data = data.drop(key, axis=1).join(dummies)
                    self.catCols.extend(dummies.keys())
                except:
                    data = data.drop(key, axis=1)
        print('[preprocessing] Found %i variables, %i numeric, %i dates, %i categorical' % (
            len(data.keys()), len(self.numCols), len(self.dateCols), len(self.catCols)))

        # Duplicates
        n_samples = len(data)
        data = data.drop_duplicates()
        diff = len(data) - n_samples
        if diff > 0:
            print('[preprocessing] Dropped %i duplicate rows' % diff)

        # Remove Anomalies
        n_nans = data.isna().sum().sum()
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
        diff = data.isna().sum().sum() - n_nans
        if diff > 0:
            print('[preprocessing] Removed %i outliers.' % diff)

        # Missing Values
        # todo check for big chunks of missing data
        data = data.replace([np.inf, -np.inf], np.nan)
        if self.missingValues == 'remove':
            data = data[data.isna().sum(axis=1) == 0]
        elif self.missingValues == 'interpolate':
            data = data.interpolate()
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif self.missingValues == 'mean':
            data = data.fillna(data.mean())
        if n_nans + diff > 0:
            print('[preprocessing] Filled %i NaN with %s' % (n_nans + diff, self.missingValues))

        # Save
        if self.outputPath:
            print('[preprocessing] Saving to: ', self.outputPath + path)
            data.to_csv(self.outputPath + path, index=self.indexCol is not None)
        return data


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

    def convert(self, data, output=None):
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
        if output:
            self.output_mean = output.mean()
            self.output_var = output.var()
            self.output_min = output.min()
            self.output_max = output.max()
            if self.type == 'normal':
                output -= self.output_mean
                output /= np.sqrt(self.output_var)
            elif self.type == 'minmax':
                output -= self.output_min
                output /= self.output_max - self.output_min
            return data, output
        else:
            return data


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


    def run(self):
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

    def featureRanking(self, args={}):
        if set(self.output) == {1, -1}:
            model = RandomForestClassifier(**args).fit(self.data, self.output)
        else:
            model = RandomForestRegressor(**args).fit(self.data, self.output)
        fig = plt.figure(figsize=[24, 16])
        ind = np.flip(np.argsort(model.feature_importances_))
        plt.bar(range(max(ind) + 1), height=model.feature_importances_)
        plt.xticks(rotation=60)
        np.save('EDA/Nonlinear Correlation/' + self.tag + '.npy', model.feature_importances_)
        fig.savefig('EDA/Nonlinear Correlation/' + self.tag + 'RF.png', format='png', dpi=300)
        plt.close()


class Modelling(object):

    def __init__(self, input, output, regression=False, classification=False):
        if regression:
            self.regression(input, output)
        if classification:
            self.classification(input, output)

    def classification(self, input, output):
        # LDA, QDA, LSVM, KNN, RBF SVM, DT, RF, ANN, LSTM, AdaBoost,
        return 0

    def regression(self, input, output):
        # Ridge, Lasso, SGD, KNN, DT, AdaBoost, GradBoost, RF, MLP, lightGBM, HistGradBoost, XG Boost, LSTM
        return 0


class Metrics(object):

    def norm_r2(ytrue, ypred, yshift):
        ytrue = ytrue.reshape((-1))
        ypred = ypred.reshape((-1))
        yshift = yshift.reshape((-1))
        # Shifted R2
        res = sum((ytrue - yshift) ** 2)
        tot = sum((yshift - np.mean(ytrue)) ** 2)
        shift_r2 = 1 - res / tot
        # Pred R2
        res = sum((ytrue - ypred) ** 2)
        tot = sum((ytrue - np.mean(ytrue)) ** 2)
        r2 = 1 - res / tot

    def r2score(ytrue, ypred):
        ytrue = ytrue.reshape((-1))
        ypred = ypred.reshape((-1))
        res = sum((ytrue - ypred) ** 2)
        tot = sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - res / tot

    def mae(ytrue, ypred):
        ytrue = ytrue.reshape((-1))
        ypred = ypred.reshape((-1))
        return np.mean(abs(ytrue - ypred))

    def mse(ytrue, ypred):
        ytrue = ytrue.reshape((-1))
        ypred = ypred.reshape((-1))
        return np.mean((ytrue - ypred) ** 2)


class Sequence(object):

    def __init__(self, back=0, forward=0, shift=0, diff='none'):
        '''
        Sequencer. Sequences and differnces data.
        Scenarios:
        - Sequenced I/O                     --> back & forward in int or list of ints
        - Sequenced input, single output    --> back: int, forward: list of ints
        - Single input, single output       --> back & forward = 1


        :param back: Int or List[int]: input indices to include
        :param forward: Int or List[int]: output indices to include
        :param shift: Int: If there is a shift between input and output
        :param diff: differencing algo, pick between 'none', 'diff', 'logdiff', 'frac' (no revert)
        '''
        if type(back) == int:
            back = np.linspace(0, back, back+1).astype('int')
            self.inputDtype = 'int'
        elif type(back) == list:
            back = np.array(back)
            self.inputDtype = 'list'
        else:
            raise ValueError('Back needs to be int or list(int)')
        if type(forward) == int:
            self.outputDtype = 'int'
            forward = np.linspace(0, forward, forward+1).astype('int')
        elif type(forward) == list:
            self.outputDtype = 'list'
            forward = np.array(forward)
        else:
            raise ValueError('Forward needs to be int or list(int)')
        self.backVec = back
        self.foreVec = forward
        self.nback = len(back)
        self.nfore = len(forward)
        self.foreRoll = np.roll(self.foreVec, 1)
        self.foreRoll[0] = 0
        self.mback = max(back) - 1
        self.mfore = max(forward) - 1
        self.shift = shift
        self.diff = diff
        self.samples = 1
        if diff != 'none':
            self.samples = 0
            self.nback -= 1
        if diff not in ['none', 'diff', 'logdiff']:
            raise ValueError('Type should be in [None, diff, logdiff, frac]')


    def convert_numpy(self, input, output, flat=False):
        if input.ndim == 1:
            input = input.reshape((-1, 1))
        if output.ndim == 1:
            output = output.reshape((-1, 1))
        samples = len(input) - self.mback - self.mfore - self.shift - 1
        features = len(input[0])
        input_sequence = np.zeros((samples, self.nback, features))
        output_sequence = np.zeros((samples, self.nfore))
        if self.diff == 'none':
            for i in range(samples):
                input_sequence[i] = input[i + self.backVec]
                output_sequence[i] = output[i - 1 + self.mback + self.shift + self.foreVec].reshape((-1))
            return input_sequence, output_sequence
        elif self.diff[-4:] == 'diff':
            if self.diff == 'logdiff':
                input = np.log(input)
                output = np.log(output)
            if (self.backVec == 0).all():
                self.backVec = np.array([0, 1])
            for i in range(samples):
                input_sequence[i] = input[i + self.backVec[1:]] - input[i + self.backVec[:-1]]
                output_sequence[i] = (output[i + self.mback + self.shift + self.foreVec] -
                                      output[i + self.mback + self.shift + self.foreRoll]).reshape((-1))
            return input_sequence, output_sequence

    def convert_pandas(self, input, output):
        assert len(input) == len(output)
        inputKeys = input.keys()
        outputKeys = output.keys()
        if self.diff == 'none':
            for lag in self.backVec:
                keys = [key + '_' + str(lag) for key in inputKeys]
                input.iloc[:, keys] = input.iloc[:, inputKeys].shift(lag)
            for shift in self.foreVec:
                keys = [key + '_' + str(shift) for key in outputKeys]
                output.iloc[:, keys] = output.iloc[:, outputKeys].shift(-shift)
        elif self.diff[-4:] == 'diff':
            for lag in self.backVec:
                keys = [key + '_' + str(lag) for key in inputKeys]
                dkeys = [key + '_d_' + str(lag) for key in inputKeys]
                input[keys] = input[inputKeys].shift(lag)
                input[dkeys] = input[inputKeys].shift(lag) - input[inputKeys]
            for shift in self.foreVec:
                keys = [key + '_' + str(shift) for key in outputKeys]
                output[keys] = output[outputKeys].shift(lag) - output[outputKeys]
        input = input.drop([key for key in input.keys() if '_0' in key], axis=1)
        output = output.drop([key for key in output.keys() if '_0' in key], axis=1)
        return input.iloc[lag:-shift], output.iloc[lag:-shift]




    def revert(self, differenced, original):
        # unsequenced integrating loop: d = np.hstack((d[0], d[0] + np.cumsum(dd)))
        if self.nfore == 1:
            differenced = differenced.reshape((-1, 1))
        output = np.zeros_like(differenced)
        if self.diff == 'logdiff':
            for i in range(self.nfore):
                output[:, i] = np.log(original[self.mback + self.foreRoll[i]:-self.foreVec[i]]) + differenced[:, i]
            return np.exp(output)
        if self.diff == 'diff':
            for i in range(self.nfore):
                output[:, i] = original[self.mback + self.foreRoll[i]:-self.foreVec[i]] + differenced[:, i]
            return output


class GridSearch(object):

    def __init__(self, model, params, cv=None, scoring=Metrics.r2score):
        self.parsed_params = []
        self.result = []
        self.model = model
        self.params = params
        if cv == None:
            self.cv = StratifiedKFold(n_splits=3)
        else:
            self.cv = cv
        self.scoring = scoring
        self._parse_params()

        if scoring == Metrics.r2score:
            self.best = [-np.inf, 0]
        else:
            self.best = [np.inf, 0]

    def _parse_params(self):
        k, v = zip(*self.params.items())
        self.parsed_params = [dict(zip(k, v)) for v in itertools.product(*self.params.values())]
        print('\n*** Grid Search ***')
        print('%i folds with %i parameter combinations, %i runs.' % (
            self.cv.n_splits,
            len(self.parsed_params),
            len(self.parsed_params) * self.cv.n_splits))


    def fit(self, input, output):
        print('\n')
        for i, param in enumerate(self.parsed_params):
            print(param)
            scoring = []
            timing = []
            for train_ind, val_ind in self.cv.split(input):
                # Start Timer
                t = time.time()

                # Split data
                xtrain, xval = input[train_ind], input[val_ind]
                ytrain, yval = output[train_ind], output[val_ind]

                # Model training
                model = sklearn.base.clone(self.model, safe=True)
                model.set_params(**param)
                model.fit(xtrain, ytrain)
                scoring.append(self.scoring(model.predict(xval), yval))
                timing.append(time.time() - t)
            if self.scoring == Metrics.r2score:
                if np.mean(scoring) - np.std(scoring) > self.best[0] - self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            else:
                if np.mean(scoring) + np.std(scoring) <= self.best[0] + self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            print('[%s] Score: %.1f \u00B1 %.1f (in %.1f seconds) (Best score so-far: %.1f \u00B1 %.1f) (%i / %i)' %
                  (datetime.now().stftime('%H:%M'), np.mean(scoring), np.std(scoring), np.mean(timing), self.best[0], self.best[1], i, len(self.parsed_params)))
            self.result.append({
                'scoring': scoring,
                'mean_score': np.mean(scoring),
                'std_score': np.std(scoring),
                'time': timing,
                'mean_time': np.mean(timing),
                'std_time': np.std(timing),
                'params': param
            })
        return pd.DataFrame(self.result)



# Objects for Hyperparameter Optimization
class LSTM(object):

    def __init__(self, dense=[1], stacked=[100, 100], bidirectional=False,
                 conv=False, conv_filters=64, conv_kernel_size=1, conv_pool_size=2, conv_seq=2,
                 dropout=False, dropout_frac=0.1,
                 regularization=1e-4, activation='tanh', loss='mse', lstm_input=None,
                 optimizers=None, epochs=50, batch_size=256, shuffle=True, early_stopping=None):
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, convolutional, Flatten
        from keras import optimizers, regularizers
        try:
            self.model = Sequential()
            if conv:
                i, j = lstm_input
                self.model.add(TimeDistributed(convolutional.Conv1D(
                    filters=conv_filters,
                    kernel_size=conv_kernel_size,
                    activation=activation
                ), input_shape=conv_input))
                self.model.add(TimeDistributed(convolutional.MaxPooling1D(pool_size=(None, int(i / conv_seq), j))))
                self.model.add(TimeDistributed(Flatten()))
            if bidirectional:
                self.model.add(Bidirectional(LSTM(stacked[0], activation=activation), input_shape=lstm_input))
                for i in range(len(stacked)-1):
                    self.model.add(LSTM(stacked[i+1], activation=activation))
            else:
                self.model.add(LSTM(stacked[0], activation=activation), input_shape=lstm_input)
                for i in range(len(stacked)-1):
                    self.model.add(LSTM(stacked[i+1], activation=activation))
            for dense_layer in dense:
                self.model.add(Dense(dense_layer))
            self.model.compile(optimizers=optimizers, loss=loss)

            self.epochs = epochs
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.callbacks = early_stopping
            return self.model
        except Exception as e:
            print(e)


    def set_param(self, params):
        self.__init__(**params)


    def fit(self, input, output):
        self.model.fit(input, output,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=self.shuffle,
                       callbacks=self.callbacks)
        return self.model


    def predict(self, input):
        return self.model.predict(input)








