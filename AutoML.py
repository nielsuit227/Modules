import os, time, itertools, re, joblib, json, pickle
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp

import sklearn
from sklearn import neural_network
from sklearn.metrics import plot_roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Priority
# Regression Validation Pipeline
# todo implement SHAP in EDA
# Add CatBoost to Modelling & get_hyper_params

# Nicety
# todo preprocessing check for big chunks of missing data
# todo multiprocessing for EDA RF, GridSearch
# todo flat=bool for sequence (needed for lightGBM)
# todo implement autoencoder for Feature Extraction

class Pipeline(object):
    def __init__(self, target,
                 missing_values='interpolate',
                 max_lags=15,
                 info_threshold=0.975,
                 max_diff=1,
                 shuffle=False,
                 n_splits=1,
                 include_output=False,
                 use_prev_data=True,
                 use_prev_mod=True,
                 shift=[0]):
        print('\n\n*** Start Amplo PM Model Builder ***\n\n')

        # Checks
        assert type(target) == str;
        assert type(shift) == list;
        assert type(info_threshold) == float;
        assert type(max_diff) == int;
        assert max_lags < 50;
        assert info_threshold > 0 and info_threshold < 1;
        assert max_diff < 5;
        assert max(shift) >= 0 and min(shift) < 50;

        # Data prep params
        self.missing_values = missing_values
        self.max_lags = max_lags
        self.info_threshold = info_threshold
        self.max_diff = max_diff
        self.include_output = include_output
        self.prev_data = use_prev_data

        # Modelling Params
        self.shuffle = shuffle
        self.prev_mod = use_prev_mod
        self.n_splits = n_splits

        # Instance initiating
        self.input = None
        self.output = None
        self.best_model = None
        self.col_keep = None
        self.best_features = None
        self.target = re.sub('[^a-zA-Z0-9 \n\.]', '_', target.lower())
        self.shift = shift
        self.catKeys = []
        self.dateKeys = []
        self.norm = Normalize(type='minmax')
        self.prep = None
        self.seq = None
        self.grid = None
        self.diff = 'none'
        self.diffOrder = 0
        self.classification = False
        self.regression = True
        self.create_dirs()

    def create_dirs(self):
        dirs = ['AutoML/', 'AutoML/Models/', 'AutoML/Data/', 'AutoML/Hyperparameter Opt/',
                'AutoML/Instances/']
        for dir in dirs:
            try:
                os.mkdir(dir)
            except:
                pass

    def fit(self, data):
        # Data preparation -- Creates/Loads All_selected, output, pp_selected, rf_selected
        self.data_preparation(data)
        # Initial Modelling
        self.initial_modelling()
        # Hyperparameter optimization
        self.grid_search()

        # Retrain and save
        if self.prev_mod and 'OptimalModel.joblib'in os.listdir('AutoML/'):
            print('[autoML] Already conducted & stored full AutoML cycle.')
        else:
            print('[autoML] Retraining with optimal parameters.')
            f = 'AutoML/Hyperparameter Opt/'
            for file in os.listdir(f):
                results = pd.read_csv(f + file)
                params = json.loads(results.iloc[0]['params'].replace("'", '"'))
                self.best_model.set_params(**params)
                self.best_model.fit(self.input, self.output)
                joblib.dump(self.best_model, 'AutoML/%s.joblib' % file[:file.find('.csv')])

            print('\n\n[autoML] Completed, score: %.4f \u00B1 %.4f' % (results.iloc[0]['mean_score'], results.iloc[0]['std_score']))

        # Validation
        self.validate()

    def predict(self, sample):
        normed = self.norm.convert(sample)
        pred = self.best_model.predict(normed)
        return self.norm.revert_output(pred)

    def data_preparation(self, data):
        files = os.listdir('AutoML/Data/')
        if self.prev_data and len(files) != 0:
            self.input = pd.read_csv('AutoML/Data/All_selected_data.csv', index_col='index')
            self.output = pd.read_csv('AutoML/Data/Output.csv', index_col='index')
            if len(set(self.output[self.target])) == 2:
                self.classification = True
                self.regression = False
        else:
            # Clean
            self.prep = Preprocessing(missingValues=self.missing_values)
            data = self.prep.clean(data)

            # Split data
            self.output = data[self.target]
            self.input = data
            if self.include_output is False:
                self.input = self.input.drop(self.target, axis=1)

            # EDA
            self.eda = ExploratoryDataAnalysis(self.input, output=self.output, folder='AutoML/')

            # Check whether is classification
            if len(set(self.output)) == 2:
                self.classification = True
                self.regression = False
            if self.classification:
                output_set = set(self.output)
                if output_set != {-1, 1}:
                    print('[autoML] WARNING: Classification labels should be {-1, 1}, AutoML changed them.')
                    self.output.loc[self.output == list(output_set)[0]] = -1
                    self.output.loc[self.output == list(output_set)[1]] = 1

            # Normalize
            if self.regression:
                self.input, self.output = self.norm.convert(self.input, output=self.output)
            elif self.classification:
                self.input = self.norm.convert(self.input)

            # Stationarity Check
            if self.max_diff != 0:
                varVec = np.zeros((self.max_diff + 1, len(self.input.keys())))
                diffData = self.input.copy(deep=True)
                for i in range(self.max_diff + 1):
                    varVec[i, :] = diffData.std()
                    diffData = diffData.diff(1)[1:]
                self.diffOrder = np.argmin(np.sum(varVec, axis=1))
                if self.diffOrder == 0:
                    self.diff = 'none'
                else:
                    self.diff = 'diff'
                    print('[autoML] Optimal Differencing order: %i' % self.diffOrder)
            else:
                self.diff = 'none'

            # Differencing, shifting, lagging
            n_features = len(self.input.keys())
            self.seq = Sequence(back=self.max_lags, forward=self.shift, diff=self.diff)
            self.input, self.output = self.seq.convert_pandas(self.input, self.output) # Double brackets keep it DataFrame
            print('[autoML] Added %i lags' % (len(self.input.keys()) - n_features))

            # Remove Colinearity
            nk = len(self.input.keys())
            norm = (self.input - self.input.mean(skipna=True, numeric_only=True)).to_numpy()
            ss = np.sqrt(np.sum(norm ** 2, axis=0))
            corr_mat = np.zeros((nk, nk))
            for i in range(nk):
                for j in range(nk):
                    if i == j:
                        continue
                    if corr_mat[i, j] == 0:
                        c = abs(np.sum(norm[:, i] * norm[:, j]) / (ss[i] * ss[j]))
                        corr_mat[i, j] = c
                        corr_mat[j, i] = c
            upper = np.triu(corr_mat)
            col_drop = self.input.keys()[np.sum(upper > self.info_threshold, axis=0) > 0].to_list()
            self.input = self.input.drop(col_drop, axis=1)
            print('[autoML] Dropped %i Co-Linear variables.' % len(col_drop))

            # Keep all
            self.col_keep = [self.input.keys().to_list()]
            self.input.to_csv('AutoML/Data/All_selected_data.csv', index_label='index')
            self.output.to_csv('AutoML/Data/Output.csv', index_label='index')

            # Keep based on PPScore
            data = self.input.copy()
            data['target'] = self.output.copy()
            pp_score = pps.predictors(data, "target")
            self.col_keep.append(pp_score['x'][pp_score['ppscore'] != 0].to_list())
            pp_data = self.input[self.col_keep[-1]]
            pp_data.to_csv('AutoML/Data/pp_selected_data.csv', index_label='index')

            # Keep based on RF
            if self.regression:
                rf = RandomForestRegressor().fit(self.input, self.output[self.target])
            elif self.classification:
                rf = RandomForestClassifier().fit(self.input, self.output[self.target])
            fi = rf.feature_importances_
            sfi = fi.sum()
            ind = np.flip(np.argsort(fi))
            ind_keep = [ind[i] for i in range(len(ind)) if fi[ind[:i]].sum() <= self.info_threshold * sfi]
            self.col_keep.append(self.input.keys()[ind_keep].to_list())
            rf_data = self.input[self.col_keep[-1]]
            rf_data.to_csv('AutoML/Data/rf_selected_data.csv', index_label='index')

            # Store to self
            json.dump(col_keep, open('AutoML/Data/Col_keep.json', 'w'))
            pickle.dump(self.prep, open('AutoML/Data/Preprocessing.pickle', 'wb'))
            pickle.dump(self.norm, open('AutoML/Data/Normalization.pickle', 'wb'))

    def initial_modelling(self):
        # Check if not already completed
        if self.prev_mod and os.path.exists('AutoML/Models/best_model.joblib'):
            self.best_model = joblib.load('AutoML/Models/best_model.joblib')
            with open('AutoML/Data/best_features.json', 'r') as f:
                self.best_features = json.load(f)
        else:
            # Load col keep
            if self.col_keep is None:
                with open('AutoML/Data/Col_keep.json', 'r') as f:
                    self.col_keep = json.load(f)

            # Initial Modelling
            init = []
            for cols in self.col_keep:
                init.append(Modelling(self.input.reindex(columns=cols), self.output.loc[:, self.target],
                                      regression=self.regression,
                                      shuffle=self.shuffle,
                                      n_splits=self.n_splits,
                                      classification=self.classification,
                                      store_folder='AutoML/Models/'))

            # Find best Dataset / Model
            if self.regression:
                data_ind = np.argmin([min(x.acc) for x in init])
                model_ind = np.argmin(init[data_ind].acc)
            elif self.classification:
                data_ind = np.argmax([max(x.acc) for x in init])
                model_ind = np.argmax(init[data_ind].acc)

            # Store
            with open('AutoML/Data/best_features.json', 'w') as f:
                json.dump(self.col_keep[data_ind], f)
            self.input = self.input[self.col_keep[data_ind]]
            self.input.to_csv('AutoML/Data/Selected.csv', index_label='index')
            self.best_model = init[data_ind].models[model_ind]
            joblib.dump(self.best_model, 'AutoML/Models/best_model.joblib')

    def get_hyper_params(self):
        print('[autoML] Getting Hyperparameters for ', type(self.best_model).__name__)
        # Parameters for both Regression / Classification
        if isinstance(self.best_model, sklearn.linear_model.Lasso) or \
            isinstance(self.best_model, sklearn.linear_model.Ridge):
            return {
                'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]
            }
        elif isinstance(self.best_model, sklearn.svm.SVC) or \
                isinstance(self.best_model, sklearn.svm.SVR):
            return {
                'gamma': [0.001, 0.01, 0.1, 0.5, 1],
                'C': [0.01, 0.1, 0.2, 0.5],
            }
        elif isinstance(self.best_model, sklearn.neighbors.KNeighborsRegressor) or \
                isinstance(self.best_model, sklearn.neighbors.KNeighborsClassifier):
            return {
                'n_neighbors': [5, 10, 25, 50],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 30, 50, 100],
                'n_jobs': [mp.cpu_count()-1],
            }
        elif isinstance(self.best_model, sklearn.neural_network._multilayer_perceptron.MLPClassifier) or \
            isinstance(self.best_model, sklearn.neural_network._multilayer_perceptron.MLPRegressor):
            return {
                'hidden_layer_sizes': [(100,), (100, 100), (100, 50), (200, 200), (200, 100), (200, 50), (50, 50, 50, 50)]
            }

        # Regressor specific hyperparameters
        elif self.regression:
            if isinstance(self.best_model, sklearn.linear_model.SGDRegressor):
                return {
                    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
                }
            elif isinstance(self.best_model, sklearn.tree.DecisionTreeRegressor):
                return {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    'max_depth': [None, 5, 10, 25, 50],
                }
            elif isinstance(self.best_model, sklearn.ensemble.AdaBoostRegressor):
                return {
                    'n_estimators': [25, 50, 100, 250],
                    'loss': ['linear', 'square', 'exponential'],
                    'learning_rate': [0.5, 0.75, 0.9, 0.95, 1]
                }
            elif isinstance(self.best_model, sklearn.ensemble.GradientBoostingRegressor):
                return {
                    'loss': ['ls', 'lad', 'huber'],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.4],
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 5, 10],
                }
            elif isinstance(self.best_model, sklearn.ensemble.HistGradientBoostingRegressor):
                return {
                    'max_iter': [100, 250],
                    'max_bins': [100, 255],
                    'loss': ['least_squares', 'least_absolute_deviation'],
                    'l2_regularization': [0.001, 0.005, 0.01, 0.05],
                    'learning_rate': [0.01, 0.1, 0.25, 0.4],
                    'max_leaf_nodes': [31, 50, 75, 150],
                    'early_stopping': [True]
                }
            elif isinstance(self.best_model, sklearn.ensemble.RandomForestRegressor):
                return {
                    'criterion': ['mse', 'mae'],
                    'max_depth': [None, 5, 10, 25, 50],
                }

        # Classification specific hyperparameters
        elif self.classification:
            if isinstance(self.best_model, sklearn.linear_model.SGDClassifier):
                return {
                    'loss': ['hinge', 'log' 'modified_huber', 'squared_hinge'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
                }
            elif isinstance(self.best_model, sklearn.tree.DecisionTreeClassifier):
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 25, 50],
                }
            elif isinstance(self.best_model, sklearn.ensemble.AdaBoostClassifier):
                return {
                    'n_estimators': [25, 50, 100, 250],
                    'learning_rate': [0.5, 0.75, 0.9, 0.95, 1]
                }
            elif isinstance(self.best_model, sklearn.ensemble.GradientBoostingClassifier):
                return {
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.4],
                    'n_estimators': [100, 300, 500],
                    'max_depth': [None, 3, 5, 10],
                }
            elif isinstance(self.best_model, sklearn.ensemble.HistGradientBoostingClassifier):
                return {
                    'max_iter': [100, 250],
                    'max_bins': [100, 255],
                    'loss': ['deviance', 'exponential'],
                    'l2_regularization': [0.001, 0.005, 0.01, 0.05],
                    'learning_rate': [0.01, 0.1, 0.25, 0.4],
                    'max_leaf_nodes': [31, 50, 75, 150],
                    'early_stopping': [True]
                }
            elif isinstance(self.best_model, sklearn.ensemble.RandomForestClassifier):
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 25, 50],
                }

        # Raise error if nothing is returned
        raise NotImplementedError('Hyperparameter tuning not implemented for ', type(self.best_model).__name__)

    def grid_search(self, model=None, params=None):
        # Optional Argument
        if model is not None:
            self.best_model = model

        # Check if Optimization is not already completed
        if self.prev_mod and type(self.best_model).__name__ + '.csv' in os.listdir('AutoML/Hyperparameter Opt/'):
            # Need to load optimal model here, need to check which hyperparameters are best :(
            score = None
            for file in os.listdir('AutoML/Hyperparameter Opt/'):
                hp_opt = pd.read_csv('AutoML/Hyperparameter Opt/' + file)
                if score is not None:
                    if hp_opt.loc[0, 'worst_case'] < score:
                        continue
                score = hp_opt.loc[0, 'worst_case']
                model = file[:file.find('.csv')]
                params = json.loads(hp_opt.loc[0, 'params'].replace("'", '"'))
            model_file = [x for x in os.listdir('AutoML') if model in x][0]
            self.best_model = joblib.load('AutoML/' + model_file)
            self.best_model.set_params(**params)
        else:
            # Get parameters
            print('[autoML] Starting Hyperparameter Optimization (%i sample, %i features)' %
                  (len(self.input), len(self.input.keys())))
            if params is None:
                params = self.get_hyper_params()

            # Set up Cross-Validation
            if self.regression:
                self.grid = GridSearch(self.best_model, params,
                                       cv=KFold(n_splits=self.n_splits),
                                       scoring=Metrics.mae)
                results = self.grid.fit(self.input, self.output)
                results['worst_case'] = results['mean_score'] + results['std_score']
                results = results.sort_values('worst_case')
            elif self.classification:
                self.grid = GridSearch(self.best_model, params,
                                       cv=StratifiedKFold(n_splits=self.n_splits),
                                       scoring=Metrics.acc)
                results = self.grid.fit(self.input, self.output)
                results['worst_case'] = results['mean_score'] - results['std_score']
                results = results.sort_values('worst_case', ascending=False)
            results.to_csv('AutoML/Hyperparameter Opt/%s.csv' % type(self.best_model).__name__, index_label='index')

    def validate(self):
        print('\n\n[autoML] Validating results for %s' % type(self.best_model).__name__)
        if not os.path.exists('AutoML/Validation/'): os.mkdir('AutoML/Validation')

        # For classification
        if self.classification:
            # Initiating
            acc = []
            prec = []
            rec = []
            spec = []
            cm = np.zeros((2, 2))
            vals = np.zeros((2))
            aucs = []
            tprs = []
            fig, ax = plt.subplots(figsize=[16, 12])
            mean_fpr = np.linspace(0, 1, 100)

            # Modelling
            cv = StratifiedKFold(n_splits=self.n_splits)
            input, output = np.array(self.input), np.array(self.output)
            for i, (t, v) in enumerate(cv.split(self.input, self.output)):
                n = len(v)
                ti, vi, to, vo = input[t], input[v], output[t].reshape((-1)), output[v].reshape((-1))
                model = sklearn.base.clone(self.best_model)
                model.fit(ti, to)
                predictions = model.predict(vi)

                # ROC Plot
                viz = plot_roc_curve(model, vi, vo, name='ROC fold {}'.format(i), alpha=0.3, ax=ax)
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

                # Metrics
                vals += np.array([(vo == 1).sum(), (vo == -1).sum()]) / self.n_splits
                tp = np.logical_and(np.sign(predictions) == 1, vo == 1).sum()
                tn = np.logical_and(np.sign(predictions) == -1, vo == -1).sum()
                fp = np.logical_and(np.sign(predictions) == 1, vo == -1).sum()
                fn = np.logical_and(np.sign(predictions) == -1, vo == 1).sum()
                acc.append((tp + tn) / n * 100)
                prec.append(tp / (tp + fp) * 100)
                rec.append(tp / (tp + fn) * 100)
                spec.append(tn / (tn + fp) * 100)
                cm += np.array([[tp, fp], [fn, tn]]) / self.n_splits

            # Plot
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#ffa62b',
                    label='Chance', alpha=.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='#2369ec',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#729ce9', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title="Receiver operating characteristic example")
            ax.legend(loc="lower right")
            fig.savefig('AutoML/Validation/ROC_%s.png' % type(model).__name__, format='png', dpi=200)
            plt.show()
            # Results
            f1 = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) for i in range(self.n_splits)]
            p = np.mean(prec)
            r = np.mean(rec)
            print('[autoML] Accuracy:        %.2f \u00B1 %.2f %%' % (np.mean(acc), np.std(acc)))
            print('[autoML] Precision:       %.2f \u00B1 %.2f %%' % (p, np.std(prec)))
            print('[autoML] Recall:          %.2f \u00B1 %.2f %%' % (r, np.std(rec)))
            print('[autoML] Specificity:     %.2f \u00B1 %.2f %%' % (np.mean(spec), np.std(spec)))
            print('[autoML] F1-score:        %.2f \u00B1 %.2f %%' % (np.mean(f1), np.std(f1)))
            print('[autoML] Confusion Matrix:')
            print('[autoML] Pred \ true  |  ISO faulty   |   ISO operational      ')
            print('[autoML]  ISO faulty  |  %s|  %s' % (('%.1f' % cm[0, 0]).ljust(13), cm[0, 1]))
            print('[autoML]  ISO oper.   |  %s|  %s' % (('%.1f' % cm[1, 0]).ljust(13), cm[1, 1]))


class Preprocessing(object):

    def __init__(self,
                 inputPath=None,
                 outputPath=None,
                 indexCol=None,
                 parseDates=False,
                 missingValues='interpolate',
                 outlierRemoval='none',
                 zScoreThreshold=4,
                 remove_constants=True
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
        self.inputPath, self.outputPath = None, None
        if inputPath:
            self.inputPath = inputPath if inputPath[-1] == '/' else inputPath + '/'
        if outputPath:
            self.outputPath = outputPath if outputPath[-1] == '/' else outputPath + '/'

        ### Algorithms
        missingValuesImplemented = ['remove_rows', 'remove_cols', 'interpolate', 'mean', 'zero']
        outlierRemovalImplemented = ['boxplot', 'zscore', 'none']
        if outlierRemoval not in outlierRemovalImplemented:
            raise ValueError("Outlier Removal Algo not implemented. Should be in " + str(outlierRemovalImplemented))
        if missingValues not in missingValuesImplemented:
            raise ValueError("Missing Values Algo not implemented. Should be in " + str(missingValuesImplemented))
        self.missingValues = missingValues
        self.outlierRemoval = outlierRemoval
        self.zScoreThreshold = zScoreThreshold
        self.parseDates = parseDates
        self.removeConstants = remove_constants

        ### Columns
        self.numCols = []
        self.catCols = []
        self.dateCols = []
        if indexCol:
            self.indexCol = re.sub('[^a-zA-Z0-9 \n\.]', '_', indexCol.lower())
        else:
            self.indexCol = None

        ### If inputPath:
        if self.inputPath:
            for file in os.listdir(inputPath):
                print('[preprocessing] %s%s' % (inputPath, file))
                data = pd.read_csv(self.inputPath + file,
                                   index_col=self.indexCol,
                                   parse_dates=self.dateCols)
                self.clean(data, file=file)


    def clean(self, data, file=None):
        print('[preprocessing] Data Cleaning Started, %i samples' % len(data))
        # Clean column names
        newKeys = {}
        for key in data.keys():
            newKeys[key] = re.sub('[^a-zA-Z0-9 \n\.]', '_', key.lower())
        data = data.rename(columns=newKeys)

        # Duplicates
        n_samples = len(data)
        data = data.drop_duplicates()
        diff = len(data) - n_samples
        if diff > 0:
            print('[preprocessing] Dropped %i duplicate rows' % diff)
        n_cols = len(data.keys())
        data = data.loc[:,~data.columns.duplicated()]
        diff = len(data.keys()) - n_cols
        if diff > 0:
            print('[preprocessing] Dropped %i duplicate columns' % diff)

        # Identify data types & convert
        missingThreshold = 0.5
        integerThreshold = 0.01
        for key in data.keys():
            # Check for numeric first
            numeric = pd.to_numeric(data[key], errors='coerce')
            if numeric.isna().sum() < len(data) * missingThreshold:
                if numeric.max() > 943920000 and self.parseDates:
                    dates = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)
                    if np.logical_and(dates.dt.date > pd.to_datetime('2000-01-01'),
                        dates.dt.date < pd.to_datetime('today')).sum() > len(data) * missingThreshold:
                        self.dateCols.append(key)
                        data[key] = dates
                        continue
                self.numCols.append(key)
                data[key] = numeric
            else:
                dates = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)
                if np.logical_and(dates.dt.date > pd.to_datetime('2000-01-01'),
                                  dates.dt.date < pd.to_datetime('today')).sum() > len(data) * missingThreshold:
                    self.dateCols.append(key)
                    data[key] = pd.to_datetime(data[key])
                    continue
                if len(set(data[key])) <= 15:
                    dummies = pd.get_dummies(data[key])
                    data = data.drop(key, axis=1).join(dummies)
                    self.catCols.extend(dummies.keys())
                else:
                    data = data.drop(key, axis=1)
        print('[preprocessing] Found %i variables, %i numeric, %i dates, %i categorical' % (
            len(data.keys()), len(self.numCols), len(self.dateCols), len(self.catCols)))

        # Drop constant columns
        if self.removeConstants:
            data = data.drop(data.keys()[data.min() == data.max()], axis=1)

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
        data = data.replace([np.inf, -np.inf], np.nan)
        if self.missingValues == 'remove_rows':
            data = data[data.isna().sum(axis=1) == 0]
        elif self.missingValues == 'remove_cols':
            data = data.loc[:, data.isna().sum(axis=0) == 0]
        elif self.missingValues == 'zero':
            data = data.fillna(0)
        elif self.missingValues == 'interpolate':
            ik = np.setdiff1d(data.keys().to_list(), self.dateCols)
            data[ik] = data[ik].interpolate(limit_direction='both')
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif self.missingValues == 'mean':
            data = data.fillna(data.mean())
        if n_nans + diff > 0:
            print('[preprocessing] Filled %i NaN with %s' % (n_nans + diff, self.missingValues))

        # Save
        print('[preprocessing] Completed, %i samples returned' % len(data))
        if self.outputPath:
            print('[preprocessing] Saving to: ', self.outputPath + file)
            data.to_csv(self.outputPath + file, index=self.indexCol is not None)
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
        self.output_mean = None
        self.output_var = None
        self.output_min = None
        self.output_max = None
        normalizationTypes = ['normal', 'minmax']
        if type not in normalizationTypes:
            raise ValueError('Type not implemented. Should be in ' + str(normalizationTypes))

    def convert(self, data, output=None):
        '''
        Actual Conversion
        :param data: Pandas DataFrame.
        :return:
        '''
        # Note Stats
        self.mean = data.mean()
        self.var = data.var()
        self.min = data.min()
        self.max = data.max()
        # Drop constants
        data = data.drop(data.keys()[data.max() == data.min()], axis=1)
        # Normalize
        if self.type == 'normal':
            data -= self.mean
            data /= np.sqrt(self.var)
        elif self.type == 'minmax':
            data -= self.min
            data /= self.max - self.min
        if output is not None:
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

    def revert(self, data):
        if self.type == 'normal':
            data *= np.sqrt(self.var)
            data += self.mean
        elif self.type == 'minmax':
            data *= self.max - self.min
            data += data.min
        return data

    def revert_output(self, output):
        if self.type == 'normal':
            assert self.output_mean != None and self.output_var != None
            output *= np.sqrt(self.output_var)
            output += self.output_mean
        elif self.type == 'minmax':
            assert self.output_min != None and self.output_max != None
            output *= self.output_max - self.output_min
            output += self.output_min
        return output


class ExploratoryDataAnalysis(object):

    def __init__(self, data, differ=0, pretag='', output=None, maxSamples=10000, seasonPeriods=[24 * 60, 7 * 24 * 60],
                 lags=60, skip_completed=True, folder=''):
        '''
        Doing all the fun EDA in an automized script :)
        :param data: Pandas Dataframe
        :param output: Pandas series of the output
        :param seasonPeriods: List of periods to check for seasonality
        :param lags: Lags for (P)ACF and
        '''
        assert isinstance(data, pd.DataFrame)

        # Register data
        self.data = data
        self.output = output

        # General settings
        self.seasonPeriods = seasonPeriods
        self.maxSamples = maxSamples        # Timeseries
        self.differ = differ                # Correlations
        self.lags = lags                    # Correlations

        # Storage settings
        self.tag = pretag
        self.folder = folder if folder[-1] == '/' else folder + '/'
        self.skip = skip_completed

        # Create dirs
        self.createDirs()
        self.run()


    def run(self):
        # Run all functions
        print('[EDA] Generating Timeplots')
        self.timeplots()
        print('[EDA] Generating Boxplots')
        self.boxplots()
        # self.seasonality()
        print('[EDA] Generating Colinearity Plots')
        self.colinearity()
        print('[EDA] Generating Diff Var Plot')
        self.differencing()
        print('[EDA] Generating ACF Plots')
        self.completeAutoCorr()
        print('[EDA] Generating PACF Plots')
        self.partialAutoCorr()
        if self.output is not None:
            print('[EDA] Generating CCF Plots')
            self.crossCorr()
            print('[EDA] Generating Feature Ranking Plot')
            self.featureRanking()
            print('[EDA] Predictive Power Score Plot')
            self.predictivePowerScore()

    def createDirs(self):
        dirs = ['EDA', 'EDA/Boxplots', 'EDA/Seasonality', 'EDA/Colinearity', 'EDA/Lags', 'EDA/Correlation',
                'EDA/Correlation/ACF', 'EDA/Correlation/PACF', 'EDA/Correlation/Cross', 'EDA/NonLinear Correlation', 'EDA/Timeplots']
        for period in self.seasonPeriods:
            dirs.append(self.folder + 'EDA/Seasonality/' + str(period))
        for dir in dirs:
            try:
                os.mkdir(self.folder + dir)
                if dir == 'EDA/Correlation':
                    file = open(self.folder + 'EDA/Correlation/Readme.txt', 'w')
                    edit = file.write(
                        'Correlation Interpretation\n\nIf the series has positive autocorrelations for a high number of lags,\nit probably needs a higher order of differencing. If the lag-1 autocorrelation\nis zero or negative, or the autocorrelations are small and patternless, then \nthe series does not need a higher order of differencing. If the lag-1 autocorrleation\nis below -0.5, the series is likely to be overdifferenced. \nOptimum level of differencing is often where the variance is lowest. ')
                    file.close()
            except:
                continue

    def boxplots(self):
        for key in self.data.keys():
            if self.tag + key + '.png' in os.listdir(self.folder + 'EDA/Boxplots/'):
                continue
            print('[EDA] Boxplot: ', key)
            fig = plt.figure(figsize=[24, 16])
            plt.boxplot(self.data[key])
            plt.title(key)
            fig.savefig(self.folder + 'EDA/Boxplots/' + self.tag + key + '.png', format='png', dpi=300)
            plt.close()

    def timeplots(self):
        matplotlib.use('Agg')
        matplotlib.rcParams['agg.path.chunksize'] = 200000
        # Undersample
        data = self.data.iloc[np.linspace(0, len(self.data) - 1, self.maxSamples).astype('int')]
        # Plot
        for key in data.keys():
            if self.tag + key + '.png' in os.listdir(self.folder + 'EDA/Timeplots/'):
                continue
            print('[EDA] Timeplot: ', key)
            fig = plt.figure(figsize=[24, 16])
            plt.plot(data.index, data[key])
            plt.title(key)
            fig.savefig(self.folder + 'EDA/Timeplots/' + self.tag + key + '.png', format='png', dpi=100)
            plt.close(fig)

    def seasonality(self):
        for key in self.data.keys():
            print('[EDA] Seasonality: ', key)
            for period in self.seasonPeriods:
                if self.tag + key + '.png' in os.listdir(self.folder + 'EDA/Seasonality/'):
                    continue
                seasonality = STL(self.data[key], period=period).fit()
                fig = plt.figure(figsize=[24, 16])
                plt.plot(range(len(self.data)), self.data[key])
                plt.plot(range(len(self.data)), seasonality)
                plt.title(key + ', period=' + str(period))
                fig.savefig(self.folder + 'EDA/Seasonality/' + self.tag + str(period)+'/'+key + '.png', format='png', dpi=300)
                plt.close()

    def colinearity(self):
        if self.tag + 'Minimum_Representation.png' in os.listdir(self.folder + 'EDA/Colinearity/'):
            pass
        threshold = 0.95
        fig = plt.figure(figsize=[24, 16])
        sns.heatmap(abs(self.data.corr()) < threshold, annot=False, cmap='Greys')
        fig.savefig(self.folder + 'EDA/Colinearity/' + self.tag + 'All_Features.png', format='png', dpi=300)
        # Minimum representation
        corr_mat = self.data.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        minimal_rep = self.data.drop(self.data[col_drop], axis=1)
        fig = plt.figure(figsize=[24, 16])
        sns.heatmap(abs(minimal_rep.corr()) < threshold, annot=False, cmap='Greys')
        fig.savefig(self.folder + 'EDA/Colinearity/' + self.tag + 'Minimum_Representation.png', format='png', dpi=300)

    def differencing(self):
        if self.tag + 'Variance.png' in os.listdir(self.folder + 'EDA/Lags/'):
            pass
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
        fig.savefig(self.folder + 'EDA/Lags/'  + self.tag + 'Variance.png', format='png', dpi=300)

    def completeAutoCorr(self):
        for i in range(self.differ):
            self.data = self.data.diff(1)[1:]
        for key in self.data.keys():
            if self.tag + key + '_differ_' + str(self.differ) + '.png' in os.listdir(self.folder + 'EDA/Correlation/ACF/'):
                continue
            print('[EDA] ACF: ', key)
            fig = plot_acf(self.data[key], fft=True)
            plt.title(key)
            fig.savefig(self.folder + 'EDA/Correlation/ACF/' + self.tag + key + '_differ_' + str(self.differ) + '.png', format='png', dpi=300)
            plt.close()

    def partialAutoCorr(self):
        for key in self.data.keys():
            if self.tag + key + '_differ_' + str(self.differ) + '.png' in os.listdir(self.folder + 'EDA/Correlation/PACF/'):
                continue

            print('[EDA] PACF: ', key)
            fig = plot_pacf(self.data[key])
            fig.savefig(self.folder + 'EDA/Correlation/PACF/' + self.tag + key + '_differ_' + str(self.differ) + '.png', format='png', dpi=300)
            plt.title(key)
            plt.close()

    def crossCorr(self):
        folder = 'EDA/Correlation/Cross/'
        for key in self.data.keys():
            if self.tag + key + '_differ_' + str(self.differ) + '.png' in os.listdir(self.folder + folder):
                continue
            print('[EDA] Cross-Correlation: ', key)
            fig = plt.figure(figsize=[24, 16])
            plt.xcorr(self.data[key], self.output, maxlags=self.lags)
            plt.title(key)
            fig.savefig(self.folder + folder + self.tag + key + '_differ_' + str(self.differ) + '.png', format='png', dpi=300)
            plt.close()

    def featureRanking(self, args={}):
        if self.tag + 'RF.png' in os.listdir(self.folder + 'EDA/Nonlinear Correlation'):
            return
        if set(self.output) == {1, -1}:
            model = RandomForestClassifier(**args).fit(self.data, self.output)
        else:
            model = RandomForestRegressor(**args).fit(self.data, self.output)
        fig = plt.figure(figsize=[24, 16])
        ind = np.flip(np.argsort(model.feature_importances_))
        plt.bar(list(self.data.keys()[ind]), height=model.feature_importances_[ind])
        plt.xticks(rotation=60, ha='right')
        np.save(self.folder + 'EDA/Nonlinear Correlation/' + self.tag + 'RF.npy', model.feature_importances_)
        fig.savefig(self.folder + 'EDA/Nonlinear Correlation/' + self.tag + 'RF.png', format='png', dpi=300)
        plt.close()

    def predictivePowerScore(self):
        if self.tag + 'Ppscore.png' in os.listdir(self.folder + 'EDA/Nonlinear Correlation'):
            return
        data = self.data.copy()
        if isinstance(self.output, pd.core.series.Series):
            data.loc[:, 'target'] = self.output
        elif isinstance(self.output, pd.DataFrame):
            data.loc[:, 'target'] = self.output.loc[:, self.output.keys()[0]]
        pp_score = pps.predictors(data, 'target')
        fig = plt.figure(figsize=[24, 16])
        sns.barplot(data=pp_score, x='x', y='ppscore')
        plt.xticks(rotation=60, ha='right')
        fig.savefig(self.folder + 'EDA/Nonlinear Correlation/' + self.tag + 'Ppscore.png', format='png', dpi=300)
        pp_score.to_csv(self.folder + 'EDA/Nonlinear Correlation/pp_score.csv')
        plt.close()


class Modelling(object):

    def __init__(self, input, output, regression=False, classification=False, shuffle=False, split=0.2, plot=False,
                 store_folder='', n_splits=3):
        self.shuffle = shuffle
        self.split = split
        self.plot = plot
        self.acc = []
        self.n_splits = n_splits
        self.store = store_folder if store_folder[-1] != '/' else store_folder[:-1]
        if regression:
            self.regression(input, output)
        if classification:
            self.classification(input, output)

    def classification(self, input, output):
        # Convert to NumPy
        input = np.array(input)
        output = np.array(output)

        # Data
        print('[modelling] Splitting data (shuffle=%s, splits=%i)' % (str(self.shuffle), self.n_splits))
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)

        # Fix plot
        if self.plot:
            plt.figure()
            plt.plot(np.array(vo), c='#FFA62B')

        # Models
        print('[modelling] Initiating all model instances')
        from sklearn import linear_model, svm, neighbors, tree, ensemble, neural_network
        from sklearn.experimental import enable_hist_gradient_boosting
        ridge = linear_model.RidgeClassifier()
        lasso = linear_model.Lasso()
        sgd = linear_model.SGDClassifier()
        svc = svm.SVC(kernel='rbf')
        knc = neighbors.KNeighborsClassifier()
        dtc = tree.DecisionTreeClassifier()
        ada = ensemble.AdaBoostClassifier()
        bag = ensemble.BaggingClassifier()
        gbc = ensemble.GradientBoostingClassifier()
        hgbc = ensemble.HistGradientBoostingClassifier()
        rfc = ensemble.RandomForestClassifier()
        mlp = neural_network.MLPClassifier()
        self.models = [ridge, lasso, sgd, svc, knc, dtc, ada, bag, gbc, hgbc, mlp]

        # Loop through models
        for master_model in self.models:

            # Time & loops through Cross-Validation
            t_start = time.time()
            v_acc = []
            t_acc = []
            for t, v in cv.split(input, output):
                ti, vi, to, vo = input[t], input[v], output[t], output[v]
                model = sklearn.base.clone(master_model)
                model.fit(ti, to)
                v_acc.append(Metrics.acc(vo, model.predict(vi)))
                t_acc.append(Metrics.acc(to, model.predict(ti)))

            # Results
            self.acc.append(np.mean(v_acc))
            joblib.dump(model, self.store + '/AutoML_IM_' + type(model).__name__ + '_mae_%.5f.joblib' % self.acc[-1])
            if self.plot:
                plt.plot(p, alpha=0.2)
            print('[modelling] %s ACC train/val: %.1f %% / %.1f %%, training time: %.1f s' %
                  (type(model).__name__.ljust(60), np.mean(t_acc) * 100, np.mean(v_acc) * 100, time.time() - t_start))
        if self.plot:
            ind = np.where(self.acc == np.min(self.acc))[0][0]
            p = self.models[ind].predict(vi)
            plt.plot(p, color='#2369ec')
            plt.title('Predictions')
            plt.legend(['True output', 'Ridge', 'Lasso', 'SGD', 'KNR', 'DTR', 'ADA', 'GBR', 'HGBR', 'MLP', 'Best'])
            plt.ylabel('Output')
            plt.xlabel('Samples')
            plt.show()
        return self

    def regression(self, input, output):
        # Convert to NumPy
        input = np.array(input)
        output = np.array(output)

        # Data
        print('[modelling] Splitting data (shuffle=%s, split=%i %%)' % (str(self.shuffle), int(self.split * 100)))
        from sklearn import model_selection
        ti, vi, to, vo = model_selection.train_test_split(input, output, test_size=self.split, shuffle=self.shuffle)

        # Fix plot
        if self.plot:
            plt.figure()
            plt.plot(np.array(vo), c='#FFA62B')

        # Models
        print('[modelling] Initiating all model instances')
        from sklearn import linear_model, svm, neighbors, tree, ensemble, neural_network
        from sklearn.experimental import enable_hist_gradient_boosting
        ridge = linear_model.Ridge()
        lasso = linear_model.Lasso()
        sgd = linear_model.SGDRegressor()
        svr = svm.SVR(kernel='rbf')
        knr = neighbors.KNeighborsRegressor()
        dtr = tree.DecisionTreeRegressor()
        ada = ensemble.AdaBoostRegressor()
        gbr = ensemble.GradientBoostingRegressor()
        hgbr = ensemble.HistGradientBoostingRegressor()
        rfr = ensemble.RandomForestRegressor()
        mlp = neural_network.MLPRegressor()
        self.models = [ridge, lasso, sgd, svr, knr, dtr, ada, gbr, hgbr, mlp]
        for model in self.models:
            t = time.time()
            model.fit(ti, to)
            tp = model.predict(ti)
            p = model.predict(vi)
            self.acc.append(Metrics.mae(vo, p))
            joblib.dump(model, self.store + '/AutoML_IM_' + type(model).__name__ + '_mae_%.5f.joblib' % self.acc[-1])
            if self.plot:
                plt.plot(p, alpha=0.2)
            print('[modelling] %s MAE train/val: %.2f %.2f, training time: %.1f s' % (type(model).__name__.ljust(60), Metrics.mae(to, tp), Metrics.mae(vo, p), time.time() - t))
        if self.plot:
            ind = np.where(self.acc == np.min(self.acc))[0][0]
            p = self.models[ind].predict(vi)
            plt.plot(p, color='#2369ec')
            plt.title('Predictions')
            plt.legend(['True output', 'Ridge', 'Lasso', 'SGD', 'KNR', 'DTR', 'ADA', 'GBR', 'HGBR', 'MLP', 'Best'])
            plt.ylabel('Output')
            plt.xlabel('Samples')
            plt.show()
        return self


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
        # if isinstance(ytrue, pd.core.series.Series) and isinstance(ypred, pd.core.series.Series):
        #     return np.mean(abs(ytrue - ypred))
        # elif isinstance(ytrue, np.ndarray) and isinstance(ypred, np.ndarray):
        #     ytrue = ytrue.reshape((-1))
        #     ypred = ypred.reshape((-1))
        #     return np.mean(abs(ytrue - ypred))
        # else:
        return np.mean(abs(np.array(ytrue).reshape((-1)) - np.array(ypred).reshape((-1))))

    def mse(ytrue, ypred):
        ytrue = ytrue.reshape((-1))
        ypred = ypred.reshape((-1))
        return np.mean((ytrue - ypred) ** 2)

    def acc(ytrue, ypred):
        return (np.array(np.sign(ytrue)).reshape((-1)) == np.array(np.sign(ypred)).reshape((-1))).sum() / len(np.array(ytrue))


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

    def convert(self, input, output, flat=False):
        assert type(input) == type(output)
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.core.series.Series):
            self.convert_pandas(input, output)
        if isinstance(input, np.ndarray):
            self.convert_numpy(input, output, flat=flat)

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

        # Check inputs
        if isinstance(input, pd.core.series.Series):
            input = input.to_frame()
        if isinstance(output, pd.core.series.Series):
            output = output.to_frame()
        assert len(input) == len(output)
        assert isinstance(input, pd.DataFrame)
        assert isinstance(output, pd.DataFrame)

        # Keys
        inputKeys = input.keys()
        outputKeys = output.keys()

        # No Differencing
        if self.diff == 'none':
            # Input
            for lag in self.backVec:
                keys = [key + '_' + str(lag) for key in inputKeys]
                input[keys] = input[inputKeys].shift(lag)

            # Output
            for shift in self.foreVec:
                keys = [key + '_' + str(shift) for key in outputKeys]
                output[keys] = output[outputKeys].shift(-shift)

        # With differencing
        elif self.diff[-4:] == 'diff':
            # Input
            for lag in self.backVec:
                # Shifted
                keys = [key + '_' + str(lag) for key in inputKeys]
                input[keys] = input[inputKeys].shift(lag)

                # Differenced
                dkeys = [key + '_d_' + str(lag) for key in inputKeys]
                input[dkeys] = input[inputKeys].shift(lag) - input[inputKeys]

            # Output
            for shift in self.foreVec:
                # Only differenced
                keys = [key + '_' + str(shift) for key in outputKeys]
                output[keys] = output[outputKeys].shift(lag) - output[outputKeys]

        # Drop _0 (same as original)
        input = input.drop([key for key in input.keys() if '_0' in key], axis=1)
        output = output.drop([key for key in output.keys() if '_0' in key], axis=1)

        # Return (first lags are NaN, last shifts are NaN
        return input.iloc[lag:-shift if shift>0 else None], output.iloc[lag:-shift if shift>0 else None]

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

        if scoring == Metrics.mae or scoring == Metrics.mse:
            self.best = [np.inf, 0]
        else:
            self.best = [-np.inf, 0]


    def _parse_params(self):
        k, v = zip(*self.params.items())
        self.parsed_params = [dict(zip(k, v)) for v in itertools.product(*self.params.values())]
        print('[GridSearch] %i folds with %i parameter combinations, %i runs.' % (
            self.cv.n_splits,
            len(self.parsed_params),
            len(self.parsed_params) * self.cv.n_splits))


    def fit(self, input, output):
        # Convert to Numpy
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.core.series.Series):
            input = np.array(input)
        if isinstance(output, pd.DataFrame) or isinstance(output, pd.core.series.Series):
            output = np.array(output).reshape((-1))

        # Loop through parameters
        for i, param in enumerate(self.parsed_params):
            print('[GridSearch] ', param)
            scoring = []
            timing = []
            for train_ind, val_ind in self.cv.split(input, output):
                # Start Timer
                t = time.time()

                # Split data
                xtrain, xval = input[train_ind], input[val_ind]
                ytrain, yval = output[train_ind], output[val_ind]

                # Model training
                model = sklearn.base.clone(self.model, safe=True)
                model.set_params(**param)
                model.fit(xtrain, ytrain)

                # Results
                scoring.append(self.scoring(model.predict(xval), yval))
                timing.append(time.time() - t)

            # Compare scores
            if scoring == Metrics.mae or scoring == Metrics.mse:
                if np.mean(scoring) + np.std(scoring) <= self.best[0] + self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            else:
                if np.mean(scoring) - np.std(scoring) > self.best[0] - self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]

            print('[GridSearch] [%s] Score: %.4f \u00B1 %.4f (in %.1f seconds) (Best score so-far: %.4f \u00B1 %.4f) (%i / %i)' %
                  (datetime.now().strftime('%H:%M'), np.mean(scoring), np.std(scoring), np.mean(timing), self.best[0], self.best[1], i + 1, len(self.parsed_params)))
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








