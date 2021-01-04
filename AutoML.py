import os, time, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold
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

    def __init__(self, type='diff', fraction=0.5, lagCutOff=25, sequence=1):
        '''
        Stationarizing class.
        Init defines the stationarization.
        Convert converts the actual in & output data
        Revert reverts the data back to the original.
        If the data needs sequencing too, first stationarize, then sequence, then revert here.


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
        standardizationTypes = ['diff', 'logdiff', 'frac']
        if type not in standardizationTypes:
            raise ValueError('Type not implemented. Should be in ' + str(standardizationTypes))
    '''
    should probably be merged with Sequence
    options:
    - sequenced input
    - sequenced output
    '''

    def convert(self, input, output):
        '''

        :param input: input data
        :param output: output data
        :param back: which samples are fed as descriptors
        :param forward: which samples do we need to predict
        :return: differenced input/output in sequence. Output (samples, back, features), input (samples, prediction)
        '''
        self.input = input
        self.output = output
        if input.ndim == 1:
            self.input = input.reshape((-1, 1))
        if output.ndim == 1:
            self.output = output.reshape((-1, 1))
        if self.type == 'diff':
            inputDiff = np.diff(input, axis=0)
            outputDiff = np.diff(output, axis=0)
        elif self.type == 'logdiff':
            if np.min(input) <= 0 and np.min(output) <= 0:
                raise ValueError('For Logarithmic Differencing, data cannot be smaller than zero')
            elif np.min(input) <= 0:
                raise Warning('Input neglected, smaller than zero.')
                inputDiff = input
                outputDiff = np.diff(np.log(output), axis=0)
            elif np.min(output) <= 0:
                raise Warning('Output neglected, smaller than zero.')
                inputDiff = np.diff(np.log(input), axis=0)
                outputDiff = output
            else:
                inputDiff = np.diff(np.log(input), axis=0)
                outputDiff = np.diff(np.log(output), axis=0)
        elif self.type == 'frac':
            inputDiff = 0
            outputDiff = output
            weights = self.getFractionalWeights(self.fraction)
            for k in range(self.lagCutOff):
                shifted = np.roll(input, k)
                shifted[:k] = 0
                inputDiff += weights[k] * shifted
        return inputDiff, outputDiff


    def revert(self, differenced, original=None, comparison=None, back=0, forward=0):
        '''
        Revert differencing, based on last convert!

        :param differenced: The differenced & sequenced output data
        :param original: The original output data
        :return: The sequenced, UNdifferenced data
        '''
        if differenced.ndim == 1:
            if self.type == 'diff':
                return np.hstack((original[0], original[:-forward] + differenced))
            elif self.type == 'logdiff':
                return np.hstack((original[0], np.exp(np.log(original[:-1]) + differenced)))
            elif self.type == 'frac':
                weights = self.getFractionalWeights(-self.fraction)
                res = 0
                for k in range(self.lagCutOff):
                    shifted = np.hstack((comparison[:k+1], differenced[k:]))
                    shifted = np.roll(shifted, k, axis=0)
                    shifted[:k] = 0
                    res += weights[k] * shifted
                return res
        else:
            reverted = np.zeros_like(differenced)
            if self.type == 'diff':
                for i in range(forward):
                    reverted[:, i] = original[back-1:-forward-1].reshape((-1)) + np.sum(differenced[:, :i+1], axis=1)
                return reverted
            elif self.type == 'logdiff':
                for i in range(forward):
                    reverted[:, i] = np.exp(np.log(original[back-1:-forward-1]) + np.sum(differenced[:, :i+1], axis=1))
                return reverted
            elif self.type == 'frac':
                raise ValueError('Fractional Integrating is incredibly hard')
                # weights = self.getFractionalWeights(-self.fraction)
                # for k in range(self.lagCutOff):
                #     shifted = np.vstack((comparison[:k + 1], differenced[k:]))
                #     shifted = np.roll(shifted, k, axis=0)
                #     shifted[:k] = 0
                #     reverted += weights[k] * shifted
            # return reverted


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

    def __init__(self, back=1, forward=1, shift=0, diff='none'):
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
            back = np.linspace(0, back-1, back).astype('int')
            self.inputDtype = 'int'
        elif type(back) == list:
            back = np.array(back)
            self.inputDtype = 'list'
        else:
            raise ValueError('Back needs to be int or list(int)')
        if type(forward) == int:
            self.outputDtype = 'int'
            forward = np.linspace(1, forward, forward).astype('int')
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
        self.mback = max(back)
        self.mfore = max(forward)
        self.shift = shift
        self.diff = diff
        self.samples = 1
        if diff != 'none':
            self.samples = 0
            self.nback -= 1
        if diff not in ['none', 'diff', 'logdiff', 'frac']:
            raise ValueError('Type should be in [None, diff, logdiff, frac]')


    def convert(self, input, output):
        if input.ndim == 1:
            input = input.reshape((-1, 1))
        if output.ndim == 1:
            output = output.reshape((-1, 1))
        samples = len(input) - self.mback - self.mfore - self.shift + self.samples
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


    def revert(self, differenced, original):
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
        for i, param in enumerate(self.parsed_params):
            print('\n', param)
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
                if np.mean(scoring) > self.best[0]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            else:
                if np.mean(scoring) <= self.best[0]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            print('Score: %.1f \u00B1 %.1f (in %.1f seconds) (Best score so-far: %.1f \u00B1 %.1f) (%i / %i)' %
                  (np.mean(scoring), np.std(scoring), np.mean(timing), self.best[0], self.best[1], i, len(self.parsed_params)))
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






