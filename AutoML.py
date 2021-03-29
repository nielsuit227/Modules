import os, time, itertools, re, joblib, json, pickle, copy, shutil
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp

from boruta import BorutaPy
import catboost
import sklearn
from sklearn import neural_network, tree, cluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

# Priority
# split EDA in regression & classification
# add report output
# add prediction script to prod env
# add cloud function script to prod env

# Nicety
# implement HalvingGridSearchCV (https://scikit-learn.org/stable/modules/grid_search.html#successive-halving-user-guide)
# implement ensembles (http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
# preprocessing check for big chunks of missing data
# multiprocessing for EDA RF, GridSearch
# flat=bool for sequence (needed for lightGBM)
# implement autoencoder for Feature Extraction
# implement transformers

class Pipeline(object):

    def __init__(self,
                 target,
                 project='',
                 mode='regression',

                 # Data Processing
                 num_cols=[],
                 date_cols=[],
                 cat_cols=[],
                 missing_values='interpolate',
                 outlier_removal='none',
                 z_score_threshold=4,
                 include_output=False,

                 # Feature Processing
                 max_lags=10,
                 max_diff=2,
                 information_threshold=0.975,

                 # Sequencing
                 sequence=False,
                 back=0,
                 forward=0,
                 shift=0,
                 diff='none',

                 # Initial Modelling
                 shuffle=False,
                 n_splits=3,
                 store_models=True,

                 # Grid Search
                 grid_search_iterations=3,

                 # Flags
                 plot_eda=None,
                 process_data=None,
                 validate_result=None):
        # Starting
        print('\n\n*** Starting Amplo AutoML - %s ***\n\n' % project)

        # Parsing input
        if len(project) == 0: self.mainDir = 'AutoML/'
        else: self.mainDir = project if project[-1] == '/' else project + '/'
        self.target = re.sub('[^a-z0-9]', '_', target.lower())

        # Checks
        assert mode == 'regression' or mode == 'classification'
        assert isinstance(target, str)
        assert isinstance(project, str)
        assert isinstance(num_cols, list)
        assert isinstance(date_cols, list)
        assert isinstance(cat_cols, list)
        assert isinstance(shift, int)
        assert isinstance(max_diff, int)
        assert max_lags < 50
        assert information_threshold > 0 and information_threshold < 1
        assert max_diff < 5

        # Params needed
        self.numCols = num_cols
        self.dateCols = date_cols
        self.catCols = cat_cols
        self.includeOutput = include_output
        self.shuffle = shuffle
        self.nSplits = n_splits
        self.gridSearchIterations = grid_search_iterations
        self.sequence = sequence
        self.plotEDA = plot_eda
        self.processData = process_data
        self.validateResults = validate_result

        # Instance initiating
        self.mode = mode
        self.version = None
        self.input = None
        self.output = None
        self.colKeep = None
        self.results = None

        # Flags
        self._setFlags()
        # Create Directories
        self._createDirs()
        # Load Version
        self._loadVersion()

        # Sub- Classes
        self.DataProcessing = DataProcessing(target=self.target, num_cols=num_cols, date_cols=date_cols,
                                             cat_cols=cat_cols, missing_values=missing_values,
                                             outlier_removal=outlier_removal, z_score_threshold=z_score_threshold,
                                             folder=self.mainDir + 'Data/', version=self.version)
        self.FeatureProcessing = FeatureProcessing(max_lags=max_lags, max_diff=max_diff,
                                                   information_threshold=information_threshold,
                                                   folder=self.mainDir + 'Features/', version=self.version)
        self.Sequence = Sequence(back=back, forward=forward, shift=shift, diff=diff)
        self.Modelling = Modelling(mode=mode, shuffle=shuffle, store_models=store_models,
                                   store_results=False, folder=self.mainDir + 'Models/')

    def _setFlags(self):
        if self.plotEDA is None:
            self.plotEDA = self._boolAsk('Make all EDA graphs?')
        if self.processData is None:
            self.processData = self._boolAsk('Process/prepare data?')
        if self.validateResults is None:
            self.validateResults = self._boolAsk('Validate results?')

    def _loadVersion(self):
        versions = os.listdir(self.mainDir + 'Production')
        if self.version is None:
            if len(versions) == 0:
                self.version = 0
            elif self.processData:
                self.version = int(len(versions))
            else:
                self.version = int(len(versions)) - 1

        # Updates changelog
        if self.processData:
            if self.version == 0:
                file = open(self.mainDir + 'changelog.txt', 'w')
                file.write('Dataset changelog. \nv0: Initial')
                file.close()
            else:
                changelog = input("Data changelog v%i:\n")
                file = open(self.mainDir + 'changelog.txt', 'a')
                file.write(('\nv%i: ' % self.version) + changelog)
                file.close()

    def _createDirs(self):
        dirs = ['', 'Data', 'Features', 'Models', 'Production', 'Validation', 'Sets']
        for dir in dirs:
            try:
                os.makedirs(self.mainDir + dir)
            except:
                continue

    def _boolAsk(self, question):
        x = input(question + ' [y / n]')
        if x.lower() == 'n' or x.lower() == 'no':
            return False
        elif x.lower() == 'y' or x.lower() == 'yes':
            return True
        else:
            print('Sorry, I did not understand. Please answer with "n" or "y"')
            return self._boolask(question)

    def _notification(self, message):
        import requests
        print(message.replace(self.mainDir[:-1], 'autoML'))
        oAuthToken = 'xoxb-1822915844353-1822989373697-zsFM6CuC6VGTxBjHUcdZHSdJ'
        url = 'https://slack.com/api/chat.postMessage'
        data = {
            "token": oAuthToken,
            "channel": "automl",
            "text": message,
            "username": "AutoML",
        }
        requests.post(url, data=data)

    def _sortResults(self, results):
        if self.mode == 'regression':
            results['worst_case'] = results['mean_score'] + results['std_score']
            results = results.sort_values('worst_case', ascending=True)
        elif self.mode == 'classification':
            results['worst_case'] = results['mean_score'] - results['std_score']
            results = results.sort_values('worst_case', ascending=False)
        return results

    def _parseJson(self, json_string):
        if isinstance(json_string, dict): return json_string
        else:
            try:
                return json.loads(json_string\
                        .replace("'", '"')\
                        .replace("True", "true")\
                        .replace("False", "false")\
                        .replace("None", "null"))
            except:
                print('[autoML] Cannot validate, imparsable JSON.')
                print(json_string)
                return json_string

    def fit(self, data):
        '''
        Fit the full autoML pipeline.
        2. (optional) Exploratory Data Analysis
        Creates a ton of plots which are helpful to improve predictions manually
        3. Data Processing
        Cleans all the data. See @DataProcessing
        4. Feature Processing
        Extracts & Selects. See @FeatureProcessing
        5. Initial Modelling
        Runs 12 off the shelf models with default parameters for all feature sets
        If Sequencing is enabled, this is where it happens, as here, the feature set is generated.
        6. Grid Search
        Optimizes the hyperparameters of the best performing models
        7. Prepare Production Files
        Nicely organises all required scripts / files to make a prediction

        @param data: DataFrame including target
        '''
        # Execute pipeline
        self._eda(data)
        self._dataProcessing(data)
        self._featureProcessing()
        self._initialModelling()
        self.gridSearch()
        # Production Env
        self._prepareProductionFiles()
        print('[autoML] Done :)')

    def _eda(self, data):
        if self.plotEDA:
            print('[autoML] Starting Exploratory Data Analysis')
            output = data[self.target]
            input = data.drop(self.target, axis=1)
            self.eda = ExploratoryDataAnalysis(input, output=output, folder=self.mainDir + 'EDA')

    def data_preparation(self, data):
        ''' DEPRECATED '''
        # Load
        if not self.prep_data and len(columns) != 0:
            print('[autoML] Loading data version %i' % self.version)
            self.input = pd.read_csv(self.mainDir + 'Data/Input_v%i.csv' % self.version, index_col='index')
            self.output = pd.read_csv(self.mainDir + 'Data/Output_v%i.csv' % self.version, index_col='index')
            self.colKeep = json.load(open(self.mainDir + 'Sets/Col_keep_v%i.json' % self.version, 'r'))
            if len(set(self.output[self.target])) == 2:
                self.classification = True
                self.regression = False
        # Execute
        else:
            # Clean
            self.prep = Preprocessing(missingValues=self.missing_values)
            data = self.prep.clean(data)

            # Split data
            self.output = data[self.target]
            self.input = data
            if self.include_output is False:
                self.input = self.input.drop(self.target, axis=1)

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
            print('[autoML] Normalizing Data.')
            norm_features = self.input.keys().to_list()
            scaler = StandardScaler()
            if self.regression:
                self.input[self.input.keys()] = scaler.fit_transform(self.input)
                output_scaler = StandardScaler()
                self.output[self.output.keys()] = output_scaler.fit_transform(self.output.to_numpy().reshape((-1, 1))).reshape((-1))
                pickle.dump(output_scaler, open(self.mainDir + '/Normalization/Output_norm_v%i.pickle' % self.version, 'wb'))
            elif self.classification:
                self.input[self.input.keys()] = scaler.fit_transform(self.input)
            self.norm = scaler  # Important for production env that scaler doesn't have a super class
            pickle.dump(scaler, open(self.mainDir + '/Normalization/Normalization_v%i.pickle' % self.version, 'wb'))
            json.dump(norm_features, open(self.mainDir + '/Normalization/norm_features_v%i.json' % self.version, 'w'))

            # Stationarity Check
            print('[autoML] Stationarity Check')
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


            # Keep all
            print('[autoML] Storing input/output')
            self.colKeep = [self.input.keys().to_list()]
            self.input.to_csv(self.mainDir + '/Data/Input_v%i.csv' % self.version, index_label='index')
            self.output.to_csv(self.mainDir + '/Data/Output_v%i.csv' % self.version, index_label='index')

            # Keep based on PPScore
            print('[autoML] Determining Features with PPS')
            data = self.input.copy()
            data['target'] = self.output.copy()
            pp_score = pps.predictors(data, "target")
            self.colKeep.append(pp_score['x'][pp_score['ppscore'] != 0].to_list())

            # Keep based on RF
            print('[autoML] Determining Features with RF')
            if self.regression:
                rf = RandomForestRegressor().fit(self.input, self.output[self.target])
            elif self.classification:
                rf = RandomForestClassifier().fit(self.input, self.output[self.target])
            fi = rf.feature_importances_
            sfi = fi.sum()
            ind = np.flip(np.argsort(fi))
            # Based on Info Threshold (97.5% of info)
            ind_keep = [ind[i] for i in range(len(ind)) if fi[ind[:i]].sum() <= self.info_threshold * sfi]
            self.col_keep.append(self.input.keys()[ind_keep].to_list())
            # Based on Info Increment (1% increments)
            ind_keep = [ind[i] for i in range(len(ind)) if fi[i] > sfi / 100]
            self.col_keep.append(self.input.keys()[ind_keep].to_list())

            # Keep based on BorutaPy -- Numerically EXHAUSTIVE
            print('[autoML] Determining Features with Boruta')
            if self.regression:
                rf = RandomForestRegressor()
            elif self.classification:
                rf = RandomForestClassifier()
            selector = BorutaPy(rf, n_estimators='auto', verbose=0)
            selector.fit(self.input.to_numpy(), self.output.to_numpy())
            self.col_keep.append(self.input.keys()[selector.support_].to_list())

            # Store to self
            print('[autoML] Storing data preparation meta data.')
            for i in range(len(self.datasets)):
                json.dump(self.col_keep[i], open(self.mainDir + 'Features/features_%i_v%i.json' % (i, self.version), 'w'))
            json.dump(self.col_keep, open(self.mainDir + 'Sets/Col_keep_v%i.json' % self.version, 'w'))

    def _dataProcessing(self, data):
        # Load if possible
        if not self.processData and os.path.exists(self.mainDir + 'Data/Cleaned_v%i.csv' % self.version):
            data = pd.read_csv(self.mainDir + 'Data/Cleaned_v%i.csv' % self.version, index_col='index')

        # Clean
        else:
            data = self.DataProcessing.clean(data)

        # Split and store in memory
        self.output = data[self.target]
        self.input = data
        if self.includeOutput is False:
            self.input = self.input.drop(self.target, axis=1)

    def _featureProcessing(self):
        # Load if possible
        if not self.processData and os.path.exists(self.mainDir + 'Data/Extracted_v%i.csv' % self.version):
            self.input = pd.read_csv(self.mainDir + 'Data/Extracted_v%i.csv' % self.version, index_col='index')
            self.colKeep = json.load(open(self.mainDir + 'Features/Sets_v%i.json' % self.version, 'r'))

        else:
            # Extract
            self.input = self.FeatureProcessing.extract(self.input, self.output)

            # Select
            self.colKeep = self.FeatureProcessing.select(self.input, self.output)

            # Store
            self.input.to_csv(self.mainDir + 'Data/Extracted_v%i.csv' % self.version, index_label='index')
            json.dump(self.colKeep, open(self.mainDir + 'Features/Sets_v%i.json' % self.version, 'w'))

    def _initialModelling(self):
        # Load existing results
        if 'Results.csv' in os.listdir(self.mainDir):
            self.results = pd.read_csv(self.mainDir + 'Results.csv')

        # Check if this version has been modelled
        if self.results is not None and \
                (self.version == self.results['data_version']).any():
            self.results = self._sortResults(self.results)

        # Run Modelling
        else:
            init = []
            for set, cols in self.colKeep.items():
                # Skip empty sets
                if len(cols) == 0:
                    print('[autoML] Skipping %s features, empty set' % set)
                else:
                    print('[autoML] Initial Modelling for %s features (%i)' % (set, len(cols)))

                    # Apply Feature Set
                    self.Modelling.dataset = set
                    # input, output = self.input.reindex(columns=cols), self.output.loc[:, self.target]
                    input, output = self.input[cols], self.output

                    # Normalize Feature Set (Done here to get one normalization file per feature set)
                    normalizeFeatures = [k for k in input.keys() if k not in self.dateCols + self.catCols]
                    scaler = StandardScaler()
                    input[normalizeFeatures] = scaler.fit_transform(input[normalizeFeatures])
                    pickle.dump(scaler, open(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (set, self.version), 'wb'))
                    if self.mode == 'regression':
                        oScaler = StandardScaler()
                        output = oScaler.fit_transform(output)
                        pickle.dump(oScaler, open(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (set, self.version), 'wb'))

                    # Sequence if necessary
                    if self.sequence:
                        input, output = self.Sequence.convert(input, output)

                    # Do the modelling
                    results = self.Modelling.fit(input, output)

                    # Add results to memory
                    results['type'] = 'Initial modelling'
                    results['data_version'] = self.version
                    if self.results is None:
                        self.results = results
                    else:
                        self.results = self.results.append(results)

            # Save results
            self.results.to_csv(self.mainDir + 'Results.csv', index=False)

    def _getHyperParams(self, model):
        # Parameters for both Regression / Classification
        if isinstance(model, sklearn.linear_model.Lasso) or \
            isinstance(model, sklearn.linear_model.Ridge) or \
                isinstance(model, sklearn.linear_model.RidgeClassifier):
            return {
                'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]
            }
        elif isinstance(model, sklearn.svm.SVC) or \
                isinstance(model, sklearn.svm.SVR):
            return {
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1],
                'C': [0.01, 0.1, 0.2, 0.5, 1, 2, 5],
            }
        elif isinstance(model, sklearn.neighbors.KNeighborsRegressor) or \
                isinstance(model, sklearn.neighbors.KNeighborsClassifier):
            return {
                'n_neighbors': [5, 10, 25, 50],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 30, 50, 100],
                'n_jobs': [mp.cpu_count()-1],
            }
        elif isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPClassifier) or \
            isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPRegressor):
            return {
                'hidden_layer_sizes': [(100,), (100, 100), (100, 50), (200, 200), (200, 100), (200, 50), (50, 50, 50, 50)],
                'learning_rate': ['adaptive', 'invscaling'],
                'alpha': [1e-4, 1e-3, 1e-5],
                'shuffle': [False],
            }

        # Regressor specific hyperparameters
        elif self.mode == 'regression':
            if isinstance(model, sklearn.linear_model.SGDRegressor):
                return {
                    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
                }
            elif isinstance(model, sklearn.tree.DecisionTreeRegressor):
                return {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    'max_depth': [None, 5, 10, 25, 50],
                }
            elif isinstance(model, sklearn.ensemble.AdaBoostRegressor):
                return {
                    'n_estimators': [25, 50, 100, 250],
                    'loss': ['linear', 'square', 'exponential'],
                    'learning_rate': [0.5, 0.75, 0.9, 0.95, 1]
                }
            elif isinstance(model, catboost.core.CatBoostRegressor):
                return {
                    'loss_function': ['MAE', 'RMSE'],
                    'iterations': [500, 1000, 2000],
                    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5],
                }
            elif isinstance(model, sklearn.ensemble.GradientBoostingRegressor):
                return {
                    'loss': ['ls', 'lad', 'huber'],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.4],
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 5, 10],
                }
            elif isinstance(model, sklearn.ensemble.HistGradientBoostingRegressor):
                return {
                    'max_iter': [100, 250],
                    'max_bins': [100, 255],
                    'loss': ['least_squares', 'least_absolute_deviation'],
                    'l2_regularization': [0.001, 0.005, 0.01, 0.05],
                    'learning_rate': [0.01, 0.1, 0.25, 0.4],
                    'max_leaf_nodes': [31, 50, 75, 150],
                    'early_stopping': [True]
                }
            elif isinstance(model, sklearn.ensemble.RandomForestRegressor):
                return {
                    'criterion': ['mse', 'mae'],
                    'max_depth': [None, 5, 10, 25, 50],
                }

        # Classification specific hyperparameters
        elif self.mode == 'classification':
            if isinstance(model, sklearn.linear_model.SGDClassifier):
                return {
                    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
                    'max_iter': [500, 1000, 1500],
                }
            elif isinstance(model, sklearn.tree.DecisionTreeClassifier):
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 25, 50],
                }
            elif isinstance(model, sklearn.ensemble.AdaBoostClassifier):
                return {
                    'n_estimators': [25, 50, 100, 250],
                    'learning_rate': [0.5, 0.75, 0.9, 0.95, 1]
                }
            elif isinstance(model, catboost.core.CatBoostClassifier):
                return {
                    'loss_function': ['Logloss', 'MultiClass'],
                    'iterations': [500, 1000, 2000],
                    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5],
                    'verbose': [0],
                }
            elif isinstance(model, sklearn.ensemble.BaggingClassifier):
                return {
                    'n_estimators': [5, 10, 15, 25, 50],
                    'max_features': [0.5, 0.75, 1.0],
                    'bootstrap': [False, True],
                    'bootstrap_features': [True, False],
                    'n_jobs': [max(mp.cpu_count() - 2, 1)],
                }
            elif isinstance(model, sklearn.ensemble.GradientBoostingClassifier):
                return {
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.4],
                    'n_estimators': [100, 300, 500],
                    'max_depth': [None, 3, 5, 10],
                }
            elif isinstance(model, sklearn.ensemble.HistGradientBoostingClassifier):
                return {
                    'max_bins': [100, 255],
                    'l2_regularization': [0.001, 0.005, 0.01, 0.05],
                    'learning_rate': [0.01, 0.1, 0.25, 0.4],
                    'max_leaf_nodes': [31, 50, 75, 150],
                    'early_stopping': [True]
                }
            elif isinstance(model, sklearn.ensemble.RandomForestClassifier):
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 25, 50],
                }

        # Raise error if nothing is returned
        raise NotImplementedError('Hyperparameter tuning not implemented for ', type(model).__name__)

    def gridSearch(self, model=None, params=None, feature_set=None):
        """
        Runs a grid search. By default, takes the self.results, and runs for the top 3 optimizations.
        There is the option to provide a model & feature_set, but both have to be provided. In this case,
        the model & data set combination will be optimized.
        :param model:
        :param params:
        :param feature_set:
        :return:
        """
        assert model is not None and feature_set is not None or model == feature_set, \
            'Model & feature_set need to be either both None or both provided.'

        # If arguments are provided
        if model is not None:

            # Organise existing results
            results = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['data_version'] == self.version,
            )]
            results = self._sortResults(results[results['dataset'] == feature_set])

            # Check if exists and load
            if ('Hyperparameter Opt' == results['type']).any():
                hyperOptResults = results[results['type'] == 'Hyperparameter Opt']
                params = self._parseJson(hyperOptResults.iloc[0]['params'])

            # Or run
            else:
                # Parameter check
                if params is None:
                    params = self._getHyperParams(model)

                # Run Grid Search
                results = self._sortResults(self._gridSearchIteration(model, params, feature_set))

                # Store results
                results['model'] = type(model).__name__
                results['data_version'] = self.version
                results['dataset'] = featureSet
                results['type'] = 'Hyperparameter Opt'
                self.results.append(results)
                results.to_csv(self.mainDir + 'Hyperparameter Opt/%s_%s.csv' %
                               (type(model).__name__, feature_set), index_label='index')
                params = results.iloc[0]['params']

            # Validate
            self.validate(model, params, feature_set)
            return

        # If arguments aren't provided
        models = self.Modelling.return_models()
        results = self._sortResults(self.results[np.logical_and(
            self.results['type'] == 'Initial modelling',
            self.results['data_version'] == self.version,
        )])
        for iteration in range(self.gridSearchIterations):
            # Grab settings
            settings = results.iloc[iteration]
            model = models[[i for i in range(len(models)) if type(models[i]).__name__ == settings['model']][0]]
            featureSet = settings['dataset']

            # Check whether exists
            modelResults = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['data_version'] == self.version,
            )]
            modelResults = self._sortResults(modelResults[modelResults['dataset'] == featureSet])

            # If exists
            if ('Hyperparameter Opt' == modelResults['type']).any():
                hyperOptRes = modelResults[modelResults['type'] == 'Hyperparameter Opt']
                params = self._parseJson(hyperOptRes.iloc[0]['params'])

            # Else run
            else:
                params = self._getHyperParams(model)
                gridSearchResults = self._sortResults(self._gridSearchIteration(model, params, featureSet))

                # Store
                gridSearchResults['model'] = type(model).__name__
                gridSearchResults['data_version'] = self.version
                gridSearchResults['dataset'] = featureSet
                gridSearchResults['type'] = 'Hyperparameter Opt'
                self.results = self.results.append(gridSearchResults)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)
                params = gridSearchResults.iloc[0]['params']

            # Validate
            if self.validateResults:
                self._validateResult(model, params, featureSet)

    def _gridSearchIteration(self, model, params, feature_set):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        print('\n[autoML] Starting Hyperparameter Optimization for %s on %s features (%i samples, %i features)' %
              (type(model).__name__, feature_set, len(self.input), len(self.colKeep[feature_set])))

        # Select data
        input = self.input[self.colKeep[feature_set]]

        # Normalize Feature Set (Done here to get one normalization file per feature set)
        normalizeFeatures = [k for k in input.keys() if k not in self.dateCols + self.catCols]
        scaler = pickle.load(open(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
        input[normalizeFeatures] = scaler.transform(input[normalizeFeatures])
        if self.mode == 'regression':
            oScaler = pickle.load(open(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (feature_set, self.version), 'rb'))
            output = oScaler.transform(output)

        # Run for regression (different Cross-Validation & worst case (MAE vs ACC))
        if self.mode == 'regression':
            gridSearch = GridSearch(model, params,
                                   cv=KFold(n_splits=self.nSplits),
                                   scoring=Metrics.mae)
            results = gridSearch.fit(input, self.output)
            results['worst_case'] = results['mean_score'] + results['std_score']
            results = results.sort_values('worst_case')

        # run for classification
        elif self.mode == 'classification':
            gridSearch = GridSearch(model, params,
                                   cv=StratifiedKFold(n_splits=self.nSplits),
                                   scoring=Metrics.acc)
            results = gridSearch.fit(input, self.output)
            results['worst_case'] = results['mean_score'] - results['std_score']
            results = results.sort_values('worst_case', ascending=False)

        return results

    def _validateResult(self, master_model, params, feature_set):
        print('[autoML] Validating results for %s (%i %s features) (%s)' % (type(master_model).__name__,
                                                len(self.colKeep[feature_set]), feature_set, params))
        if not os.path.exists(self.mainDir + 'Validation/'): os.mkdir(self.mainDir + 'Validation/')

        # For Regression
        if self.mode == 'regression':

            # Cross-Validation Plots
            fig, ax = plt.subplots(round(self.nSplits / 2), 2, sharex=True, sharey=True)
            fig.suptitle('%i-Fold Cross Validated Predictions - %s' % (self.nSplits, type(master_model).__name__))

            # Initialize iterables
            mae = []
            cv = KFold(n_splits=self.nSplits, shuffle=self.shuffle)
            input, output = np.array(self.input[self.colKeep[feature_set]]), np.array(self.output)

            # Cross Validate
            for i, (t, v) in enumerate(cv.split(input, output)):
                ti, vi, to, vo = input[t], input[v], output[t].reshape((-1)), output[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(ti, to)
                predictions = model.predict(vi).reshape((-1))

                # Metrics
                mae.append(Metrics.mae(vo, predictions))

                # Plot
                ax[i // 2][i % 2].set_title('Fold-%i' % i)
                ax[i // 2][i % 2].plot(vo, color='#2369ec')
                ax[i // 2][i % 2].plot(predictions, color='#ffa62b', alpha=0.4)

            # Print & Finish plot
            print('[autoML] MAE:        %.2f \u00B1 %.2f' % (np.mean(mae), np.std(mae)))
            ax[i // 2][i % 2].legend(['Output', 'Prediction'])
            plt.show()

        # For classification
        if self.mode == 'classification':
            # Initiating
            acc = []
            prec = []
            rec = []
            spec = []
            aucs = []
            tprs = []
            cm = np.zeros((2, 2))
            mean_fpr = np.linspace(0, 1, 100)

            # Modelling
            cv = StratifiedKFold(n_splits=self.nSplits)
            input, output = np.array(self.input[self.colKeep[feature_set]]), np.array(self.output)
            for i, (t, v) in enumerate(cv.split(input, output)):
                n = len(v)
                ti, vi, to, vo = input[t], input[v], output[t].reshape((-1)), output[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(ti, to)
                predictions = model.predict(vi).reshape((-1))

                # Metrics
                tp = np.logical_and(np.sign(predictions) == 1, vo == 1).sum()
                tn = np.logical_and(np.sign(predictions) == -1, vo == -1).sum()
                fp = np.logical_and(np.sign(predictions) == 1, vo == -1).sum()
                fn = np.logical_and(np.sign(predictions) == -1, vo == 1).sum()
                acc.append((tp + tn) / n * 100)
                prec.append(tp / (tp + fp) * 100)
                rec.append(tp / (tp + fn) * 100)
                spec.append(tn / (tn + fp) * 100)
                cm += np.array([[tp, fp], [fn, tn]]) / self.nSplits

            # Results
            f1 = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) for i in range(self.nSplits)]
            p = np.mean(prec)
            r = np.mean(rec)
            print('[autoML] Accuracy:        %.2f \u00B1 %.2f %%' % (np.mean(acc), np.std(acc)))
            print('[autoML] Precision:       %.2f \u00B1 %.2f %%' % (p, np.std(prec)))
            print('[autoML] Recall:          %.2f \u00B1 %.2f %%' % (r, np.std(rec)))
            print('[autoML] Specificity:     %.2f \u00B1 %.2f %%' % (np.mean(spec), np.std(spec)))
            print('[autoML] F1-score:        %.2f \u00B1 %.2f %%' % (np.mean(f1), np.std(f1)))
            print('[autoML] Confusion Matrix:')
            print('[autoML] Pred \ true  |  ISO faulty   |   ISO operational      ')
            print('[autoML]  ISO faulty  |  %s|  %.1f' % (('%.1f' % cm[0, 0]).ljust(13), cm[0, 1]))
            print('[autoML]  ISO oper.   |  %s|  %.1f' % (('%.1f' % cm[1, 0]).ljust(13), cm[1, 1]))

            # Check whether plot is possible
            if isinstance(model, sklearn.linear_model.Lasso) or isinstance(model, sklearn.linear_model.Ridge):
                return

            # Plot ROC
            fig, ax = plt.subplots(figsize=[12, 8])
            viz = plot_roc_curve(model, vi, vo, name='ROC fold {}'.format(i + 1), alpha=0.3, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            # Adjust plots
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
                   title="ROC Curve - %s" % type(master_model).__name__)
            ax.legend(loc="lower right")
            fig.savefig(self.mainDir + 'Validation/ROC_%s.png' % type(model).__name__, format='png', dpi=200)

    def _prepareProductionFiles(self, model=None, feature_set=None):
        if not os.path.exists(self.mainDir + 'Production/v%i/' % self.version):
            os.mkdir(self.mainDir + 'Production/v%i/' % self.version)
        # Get sorted results for this data version
        results = self._sortResults(self.results[self.results['data_version'] == self.version])

        # In the case args are provided
        if model is not None and feature_set is not None:
            results = self._sortResults(results[np.logical_and(results['model'] == model, results['dataset'] == feature_set)])
            params = self._parseJson(results.iloc[0]['params'])

        # Otherwise find best
        else:
            model = results.iloc[0]['model']
            feature_set = results.iloc[0]['dataset']
            params = results.iloc[0]['params']
            if isinstance(params, str):
                params = self._parseJson(params)
        self._notification('[%s] Preparing Production Env Files for %s, feature set %s' %
                          (self.mainDir[:-1], model, feature_set))
        if self.mode == 'classification':
            self._notification('[%s] Accuracy: %.2f \u00B1 %.2f' %
                              (self.mainDir[:-1], results.iloc[0]['mean_score'], results.iloc[0]['std_score']))
        elif self.mode == 'regression':
            self._notification('[%s] Mean Absolute Error: %.2f \u00B1 %.2f' %
                              (self.mainDir[:-1], results.iloc[0]['mean_score'], results.iloc[0]['std_score']))

        # Copy Features & Normalization & Norm Features
        json.dump(self.colKeep[feature_set], open(self.mainDir + 'Production/v%i/Features.json' % self.version, 'w'))
        shutil.copy(self.mainDir + 'Features/Scaler_%s_%i.pickle' % (feature_set, self.version),
                    self.mainDir + 'Production/v%i/Scaler.json' % self.version)
        if self.mode == 'regression':
            shutil.copy(self.mainDir + 'Features/OScaler_%s_%i.pickle' % (feature_set, self.version),
                        self.mainDir + 'Production/v%i/OScaler.pickle' % self.version)

        # Model
        model = [mod for mod in self.Modelling.return_models() if type(mod).__name__ == model][0]
        model.set_params(**params)
        model.fit(self.input[self.colKeep[feature_set]], self.output)
        joblib.dump(model, self.mainDir + 'Production/v%i/Model.joblib' % self.version)

        # Predict function
        f = open(self.mainDir + 'Production/Predict.py', 'w')
        f.write(self.returnPredictFunc())
        f.close()
        
        # Pipeline
        pickle.dump(self, open(self.mainDir + 'Production/Pipeline.pickle', 'wb'))
        return

    def predict(self, model, features, scaler, sample):
        # todo asserts for sample
        return 'Not implemented'

    def returnPredictFunc(self):
        return '''import pandas as pd
import numpy as np
import struct, re


def decode(data, dbc):
    for key in data.keys():
        data = data.rename(columns={key: key.replace(' ', '')})
    dec_list = []
    float_inds = [1031, 1002, 997, 873, 934, 868, 1313, 1282, 1281, 1038, 1037, 1036, 1035, 1033, 1032, 1030, 1029,
                  1028, 1026, 1070, 1069, 1068, 1067, 1065, 1064, 1063, 1063, 1061, 1060, 1059, 1058]
    for j in range(9000):
        row = data.iloc[j]
        row_id = int(row['ID'], 0)
        can_bytes = bytes.fromhex(row['data'].strip()[2:])[::-1]

        # Skip legacy packages
        if row_id not in [265, 280, 769, 808, 866, 870, 871, 877, 879, 881, 885, 889, 891, 933, 943, 944, 945,
                          1025, 1026, 1035, 1057, 1058, 1063, 1067, 1889]:
            continue

        # Decode
        try:
            decoded = dbc.decode_message(row_id, can_bytes)
            decoded['ts'] = row['Recvtime']
            # Check floats
            if row_id in float_inds:
                if len(dbc.get_message_by_frame_id(row_id).signals) == 2:
                    x, y = struct.unpack('<ff', can_bytes)
                    decoded[list(decoded.keys())[0]] = x
                    decoded[list(decoded.keys())[1]] = y
                else:
                    signal = [s for s in dbc.get_message_by_frame_id(row_id).signals if s.length == 32][0]
                    x, y = struct.unpack('<ff', can_bytes)
                    if signal.start == 32:
                        decoded[signal.name] = y
                    elif signal.start == 0:
                        decoded[signal.name] = x
                # Store
                dec_list.append(decoded)
        except:
            # Bare exception is no issue. Exception is triggered for legacy messages.
            pass
    # Store
    decoded_data = pd.DataFrame(dec_list)
    return decoded_data


def preprocess(data):
    # Convert timestamps
    data['ts'] = pd.to_datetime(data['ts'])
    data = data.set_index('ts')
    # Drop duplicates
    data = data.drop_duplicates()
    data = data.loc[:, ~data.columns.duplicated()]
    # Merge rows
    new_data = pd.DataFrame(columns=[], index=data.index.drop_duplicates(keep='first'))
    for key in data.keys():
        key_series = pd.Series(data[~data[key].isna()][key])
        new_data[key] = key_series[~key_series.index.duplicated()]
    data = new_data
    # Cat cols
    cat_cols = ['ControlPilotState', 'ChargeState', 'CCSState', 'CCSStateNext', 'CCSShutdownCode',
                'CCSReinitErrorCode', 'ChargerErrorEvent1', 'ChargerErrorEvent2', 'ChargerErrorEvent3',
                'ChargerErrorEvent4', 'RFIDSwipeStatus']
    for key in cat_cols:
        if key in data.keys():
            dummies = pd.get_dummies(data[key], prefix=key).replace(0, 1)
            data = data.drop(key, axis=1).join(dummies)
    # Re-sample
    data = data.resample('ms').interpolate(limit_direction='both')
    data = data.resample('s').asfreq()
    # Cleaning Keys
    new_keys = {}
    for key in data.keys():
        new_keys[key] = re.sub('[^a-zA-Z0-9]', '_', key.lower())
    data = data.rename(columns=new_keys)
    del new_keys
    return data


class Predict(object):

    def __init__(self):
        self.version = 'v0.1.2'

    def predict(self, model, features, all_features, normalization, dbc, data):
        # Decode
        data = decode(data, dbc)

        # Convert to timeseries & preprocess
        data = preprocess(data)

        # Normalize
        for key in [x for x in all_features if x not in list(data.keys())]:
            data.loc[:, key] = np.zeros(len(data))
        data = data[all_features]
        data[data.keys()] = normalization.transform(data)

        # Select features
        for key in [x for x in features if x not in list(data.keys())]:
            data.loc[:, key] = np.zeros(len(data))
        data = data[features]

        # Make prediction
        predictions = model.predict_proba(data.iloc[1:])[:, 1]
        return sum(predictions) / len(predictions) * 100
'''


class DataProcessing(object):

    def __init__(self,
                 target=None,
                 num_cols=None,
                 date_cols=None,
                 cat_cols=None,
                 missing_values='interpolate',
                 outlier_removal='none',
                 z_score_threshold=4,
                 folder='',
                 version=1,
                 mode='regression',
                 ):
        '''
        Preprocessing Class. Deals with Outliers, Missing Values, duplicate rows, data types (floats, categorical and dates),
        NaN, Infs.

        :param target: Column name of target variable
        :param num_cols: Numerical columns, all parsed to integers and floats
        :param dates_cols: Date columns, all parsed to pd.datetime format
        :param cat_cols: Categorical Columns. Currently all one-hot encoded.
        :param missing_values: How to deal with missing values ('remove', 'interpolate' or 'mean')
        :param outlier_removal: How to deal with outliers ('boxplot', 'zscore' or 'none')
        :param zScore_threshold: If outlierRemoval='zscore', the threshold is adaptable, default=4.
        :param folder: Directory for storing the output files
        :param version: Versioning the output files
        '''
        # Parameters
        self.folder = folder if len(folder) == 0 or folder[-1] == '/' else folder + '/'
        self.version = version
        self.target = re.sub('[^a-z0-9]', '_', target.lower())
        self.mode = mode
        self.numCols = [re.sub('[^a-z0-9]', '_', nc.lower()) for nc in num_cols] if num_cols is not None else []
        self.catCols = [re.sub('[^a-z0-9]', '_', cc.lower()) for cc in cat_cols] if cat_cols is not None else []
        self.dateCols = [re.sub('[^a-z0-9]', '_', dc.lower()) for dc in date_cols]  if date_cols is not None else []
        if self.target in self.numCols:
            self.numCols.remove(self.target)

        # Variables
        self.scaler = None
        self.oScaler = None

        ### Algorithms
        missingValuesImplemented = ['remove_rows', 'remove_cols', 'interpolate', 'mean', 'zero']
        outlierRemovalImplemented = ['boxplot', 'zscore', 'none']
        if outlier_removal not in outlierRemovalImplemented:
            raise ValueError("Outlier Removal Algorithm not implemented. Should be in " + str(outlierRemovalImplemented))
        if missing_values not in missingValuesImplemented:
            raise ValueError("Missing Values Algorithm not implemented. Should be in " + str(missingValuesImplemented))
        self.missingValues = missing_values
        self.outlierRemoval = outlier_removal
        self.zScoreThreshold = z_score_threshold


    def clean(self, data):
        print('[Data] Data Cleaning Started, (%i x %i) samples' % (len(data.keys()), len(data)))
        if len(data[self.target].unique()) == 2: self.mode = 'classification'

        # Clean
        data = self._cleanKeys(data)
        data = self._convertDataTypes(data)
        data = self._removeDuplicates(data)
        data = self._removeOutliers(data)
        data = self._removeMissingValues(data)
        data = self._removeConstants(data)
        # data = self._normalize(data) --> [Deprecated] Moved right before modelling

        # Finish
        self._store(data)
        print('[Data] Processing completed, (%i x %i) samples returned' % (len(data), len(data.keys())))
        return data

    def _cleanKeys(self, data):
        newKeys = {}
        for key in data.keys():
            newKeys[key] = re.sub('[^a-zA-Z0-9 \n\.]', '_', key.lower())
        data = data.rename(columns=newKeys)
        return data

    def _convertDataTypes(self, data):
        print('[Data] Converted %i numerical, %i categorical and %i date columns' %
              (len(self.numCols), len(self.catCols), len(self.dateCols)))
        for key in self.dateCols:
            data[key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)
        for key in self.catCols:
            dummies = pd.get_dummies(data[key])
            for dummy_key in dummies.keys():
                dummies = dummies.rename(columns={dummy_key: key + '_' + str(dummy_key)})
            data = data.drop(key, axis=1).join(dummies)
        for key in [key for key in data.keys() if key not in self.dateCols and key not in self.catCols]:
            data[key] = pd.to_numeric(data[key], errors='coerce')
        data[self.target] = pd.to_numeric(data[self.target], errors='coerce')
        return data

    def _removeDuplicates(self, data):
        n_rows, n_cols = len(data), len(data.keys())
        data = data.drop_duplicates()
        print('[Data] Dropped %i duplicate rows' % (n_rows - len(data)))
        data = data.loc[:, ~data.columns.duplicated()]
        print('[Data] Dropped %i duplicate columns' % (n_cols - len(data.keys())))
        return data

    def _removeConstants(self, data):
        nCols = len(data.keys())
        data = data.drop(data.keys()[data.min() == data.max()], axis=1)
        print('[Data] Removed %i constant columns' % (nCols - len(data.keys())))
        return data

    def _removeOutliers(self, data):
        n_nans = data.isna().sum().sum()
        if self.outlierRemoval == 'boxplot':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            for key in Q1.keys():
                data.loc[data[key] < Q1[key] - 1.5 * (Q3[key] - Q1[key]), key] = np.nan
                data.loc[data[key] > Q3[key] + 1.5 * (Q3[key] - Q1[key]), key] = np.nan
        elif self.outlierRemoval == 'zscore':
            Zscore = (data - data.mean(skipna=True, numeric_only=True)) \
                     / np.sqrt(data.var(skipna=True, numeric_only=True))
            data[Zscore > self.zScoreThreshold] = np.nan
        print('[Data] Removed %i outliers.' % (data.isna().sum().sum() - n_nans))
        return data

    def _removeMissingValues(self, data):
        data = data.replace([np.inf, -np.inf], np.nan)
        n_nans = data.isna().sum().sum()
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
        if n_nans > 0:
            print('[Data] Filled %i (%.1f %%) missing values with %s' % (n_nans,
                  n_nans * 100 / len(data) / len(data.keys()), self.missingValues))
        return data

    def _normalize(self, data):
        # Organise features that need normalizing, some where deleted
        features = [key for key in data.keys() if key not in self.catCols + self.dateCols + [self.target]]

        # Normalize
        self.scaler = StandardScaler()
        data[features] = self.scaler.fit_transform(data[features])

        # Normalize output for regression
        if self.mode == 'regression':
            self.oScaler = StandardScaler()
            data[self.target] = self.oScaler.fit_transform(data[self.target])
        return data

    def _store(self, data):
        # Store cleaned data
        data.to_csv(self.folder + 'Cleaned_v%i.csv' % self.version, index_label='index')
        # Store normalization
        pickle.dump(self.scaler, open(self.folder + 'Normalizer_v%i.pickle' % self.version, 'wb'))
        if self.mode == 'r':
            pickle.dump(self.oScaler, open(self.folder + 'Output_norm_v%i.pickle' % self.version, 'wb'))

class FeatureProcessing(object):

    def __init__(self,
                 max_lags=10,
                 max_diff=2,
                 information_threshold=0.975,
                 folder='',
                 version=''):
        self.input = None
        self.output = None
        self.model = None
        self.mode = None
        self.threshold = None
        # Register
        self.baseScore= {}
        self.colinearFeatures = None
        self.crossFeatures = None
        self.kMeansFeatures = None
        self.diffFeatures = None
        self.laggedFeatures = None
        # Parameters
        self.maxLags = max_lags
        self.maxDiff = max_diff
        self.informationThreshold = information_threshold
        self.folder = folder if folder == '' or folder[-1] == '/' else folder + '/'
        self.version = version

    def extract(self, inputFrame, outputFrame):
        self._cleanAndSet(inputFrame, outputFrame)
        # Manipulate features
        self._removeColinearity()
        self._addCrossFeatures()
        self._addDiffFeatures()
        self._addKMeansFeatures()
        self._calcBaseline()
        self._addLaggedFeatures()
        return self.input

    def _cleanAndSet(self, inputFrame, outputFrame):
        assert isinstance(inputFrame, pd.DataFrame), 'Input supports only Pandas DataFrame'
        assert isinstance(outputFrame, pd.Series), 'Output supports only Pandas Series'
        if len(outputFrame.unique()) == 2:
            self.model = tree.DecisionTreeClassifier(max_depth=3)
            self.mode = 'classification'
        else:
            self.model = tree.DecisionTreeRegressor(max_depth=3)
            self.mode = 'regression'
        # Bit of necessary data cleaning (shouldn't change anything)
        self.input = inputFrame.replace([np.inf, -np.inf], 0).fillna(0).reset_index(drop=True)
        self.output = outputFrame.replace([np.inf, -np.inf], 0).fillna(0).reset_index(drop=True)

    def _calcBaseline(self):
        '''
        Calculates baseline for correlation, method same for all functions.
        '''
        # Check if not already executed
        if os.path.exists(self.folder + 'BaseScores_v%i.json' % self.version):
            print('[Features] Loading Baseline')
            self.baseScore = json.load(open(self.folder + 'BaseScores_v%i.json' % self.version, 'r'))
            return

        # Calculate scores
        print('[Features] Calculating baseline feature importance v%i' % self.version)
        for key in self.input.keys():
            m = copy.copy(self.model)
            m.fit(self.input[[key]], self.output)
            self.baseScore[key] = m.score(self.input[[key]], self.output)
        json.dump(self.baseScore, open(self.folder + 'BaseScores_v%i.json' % self.version, 'w'))

    def _removeColinearity(self):
        '''
        Calculates the Pearson Correlation Coefficient for all input features.
        Those higher than the information threshold are linearly codependent (i.e., describable by y = a x + b)
        These features add little to no information and are therefore removed.
        '''
        # Check if not already executed
        if os.path.exists(self.folder + 'Colinear_v%i.json' % self.version):
            print('[Features] Loading Colinear features')
            self.colinearFeatures = json.load(open(self.folder + 'Colinear_v%i.json' % self.version, 'r'))

        # Else run
        else:
            print('[Features] Analysing colinearity')
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
            self.colinearFeatures = self.input.keys()[np.sum(upper > self.informationThreshold, axis=0) > 0].to_list()

        # Save & Store
        json.dump(self.colinearFeatures, open(self.folder + 'Colinear_v%i.json' % self.version, 'w'))
        self.input = self.input.drop(self.colinearFeatures, axis=1)
        print('[Features] Removed %i Co-Linear features (%.3f %% threshold)' % (len(self.colinearFeatures), self.informationThreshold))

    def _addCrossFeatures(self):
        '''
        Calculates cross-feature features with m and multiplication.
        Should be limited to say ~500.000 features (runs about 100-150 features / second)
        '''
        # Check if not already executed
        if os.path.exists(self.folder + 'crossFeatures_v%i.json' % self.version):
            self.crossFeatures = json.load(open(self.folder + 'crossFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i cross features' % len(self.crossFeatures))

        # Else, execute
        else:
            print('[Features] Analysing cross features')
            # Make division todo List
            keys = self.input.keys()
            divList = []
            for key in keys:
                divList += [(key, k) for k in keys if k != key]

            # Make multiplication todo list
            multiList = []
            for i, keyA in enumerate(keys):
                for j, keyB in enumerate(keys):
                    if j >= i:
                        continue
                    multiList.append((keyA, keyB))

            # Analyse scores
            scores = {}
            for keyA, keyB in tqdm(divList):
                feature = self.input[[keyA]] / self.input[[keyB]]
                feature = feature.replace([np.inf, -np.inf], 0).fillna(0)
                m = copy.copy(self.model)
                m.fit(feature, self.output)
                scores[keyA + '__d__' + keyB] = m.score(feature, self.output)
            for keyA, keyB in tqdm(multiList):
                feature = self.input[[keyA]] * self.input[[keyB]]
                feature = feature.replace([np.inf, -np.inf], 0).fillna(0)
                m = copy.copy(self.model)
                m.fit(feature, self.output)
                scores[keyA + '__x__' + keyB] = m.score(feature, self.output)

            # Select valuable features
            scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
            items = min(250, sum(score > 0.1 for score in scores.values()))
            self.crossFeatures = [k for k, v in list(scores.items())[:items] if v > 0.1]

        # Split and Add
        multiFeatures = [k for k in self.crossFeatures if '__x__' in k]
        divFeatures = [k for k in self.crossFeatures if '__d__' in k]
        newFeatures = []
        for k in multiFeatures:
            keyA, keyB = k.split('__x__')
            feature = self.input[keyA] * self.input[keyB]
            feature = feature.astype('float32').replace([np.inf, -np.inf], 0).fillna(0)
            self.input[k] = feature
        for k in divFeatures:
            keyA, keyB = k.split('__d__')
            feature = self.input[keyA] / self.input[keyB]
            feature = feature.astype('float32').replace([np.inf, -np.inf], 0).fillna(0)
            self.input[k] = feature

        # Store
        json.dump(self.crossFeatures, open(self.folder + 'crossFeatures_v%i.json' % self.version, 'w'))
        print('[Features] Added %i cross features' % len(self.crossFeatures))

    def _addKMeansFeatures(self):
        '''
        Analyses the correlation of k-means features.
        k-means is a clustering algorithm which clusters the data.
        The distance to each cluster is then analysed.
        '''
        # Check if not exist
        if os.path.exists(self.folder + 'K-MeansFeatures_v%i.json' % self.version):
            # Load features and cluster size
            self.kMeansFeatures = json.load(open(self.folder + 'K-MeansFeatures_v%i.json' % self.version, 'r'))
            clusters = self.kMeansFeatures[0]
            clusters = int(clusters[clusters.rfind('_'):])
            print('[Features] Loaded %i K-Means features' % len(self.kMeansFeatures))

            # Calculate distances
            kmeans = cluster.MiniBatchKMeans(n_clusters=clusters)
            columnNames = ['dist_c_%i_%i' % (i, clusters) for i in range(clusters)]
            distances = pd.DataFrame(columns=columnNames, data=kmeans.fit_transform(self.input))

            # Add them
            for key in self.kMeansFeatures:
                self.input[key] = distances[key]

        # If not executed, analyse all
        else:
            print('[Features] Calculating and Analysing K-Means features')
            # Fit K-Means and calculate distances
            clusters = min(max(int(np.log10(len(self.input)) * 8), 8), len(self.input.keys()))
            kmeans = cluster.MiniBatchKMeans(n_clusters=clusters)
            columnNames = ['dist_c_%i_%i' % (i, clusters) for i in range(clusters)]
            distances = pd.DataFrame(columns=columnNames, data=kmeans.fit_transform(self.input))

            # Analyse correlation
            scores = {}
            for key in tqdm(distances.keys()):
                m = copy.copy(self.model)
                m.fit(distances[[key]], self.output)
                scores[key] = m.score(distances[[key]], self.output)

            # Add the valuable features
            self.kMeansFeatures = [k for k, v in scores.items() if v > 0.1]
            for k in self.kMeansFeatures:
                self.input[k] = distances[k]
            json.dump(self.kMeansFeatures, open(self.folder + 'crossFeatures.json', 'w'))
            print('[Features] Added %i K-Means features (%i clusters)' % (len(self.kMeansFeatures), clusters))

    def _addDiffFeatures(self):
        '''
        Analyses whether the differenced signal of the data should be included.
        '''

        # Check if we're allowed
        if self.maxDiff == 0:
            print('[Features] Differenced features skipped, max diff = 0')
            return

        # Check if exist
        if os.path.exists(self.folder + 'diffFeatures_v%i.json' % self.version):
            self.diffFeatures = json.load(open(self.folder + 'diffFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i differenced features' % len(self.diffFeatures))

        # If not exist, execute
        else:
            print('[Features] Analysing differenced features')
            # Copy data so we can diff without altering original data
            keys = self.input.keys()
            diffInput = copy.copy(self.input)

            # Calculate scores
            scores = {}
            for diff in tqdm(range(1, self.maxDiff+1)):
                diffInput = diffInput.diff().fillna(0)
                for key in keys:
                    m = copy.copy(self.model)
                    m.fit(diffInput[key], diffOutput)
                    score[key + '__diff__%i' % diff] = m.score(diffInput[key], diffOutput)

            # Select the valuable features
            self.diffFeatures = [k for k, v in scores.items() if v > 0.1]
            print('[Features] Added %i differenced features' % len(self.diffFeatures))

        # Add Differenced Features
        for k in self.diffFeatures:
            key, diff = k.split('__diff__')
            feature = self.input[key]
            for i in range(1, diff):
                feature = feature.diff().fillna(0)
            self.input[k] = feature
        json.dump(self.diffFeatures, open(self.folder + 'diffFeatures_v%i.json' % self.version, 'w'))

    def _addLaggedFeatures(self):
        '''
        Analyses the correlation of lagged features (value of sensor_x at t-1 to predict target at t)
        '''
        # Check if allowed
        if self.maxLags == 0:
            print('[Features] Lagged features skipped, max lags = 0')
            return

        # Check if exists
        if os.path.exists(self.folder + 'laggedFeatures_v%i.json' % self.version):
            self.laggedFeatures = json.load(open(self.folder + 'laggedFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i lagged features' % len(self.laggedFeatures))

        # Else execute
        else:
            print('[Features] Analysing lagged features')
            keys = self.input.keys()
            scores = {}
            for lag in tqdm(range(1, self.maxLags)):
                for key in keys:
                    m = copy.copy(self.model)
                    m.fit(self.input[[key]][:-lag], self.output[lag:])
                    scores[key + '__lag__%i' % lag] = m.score(self.input[[key]][:-lag], self.output[lag:])

            # Select
            scores = {k: v - self.baseScore[k[:k.find('__lag__')]] for k, v in sorted(scores.items(),
                                              key=lambda k: k[1] - self.baseScore[k[0][:k[0].find('__lag__')]], reverse=True)}
            items = min(250, sum(v > 0.1 for v in scores.values()))
            self.laggedFeatures = [k for k, v in list(scores.items())[:items] if v > self.baseScore[k[:k.find('__lag__')]]]
            print('[Features] Added %i lagged features' % len(self.laggedFeatures))

        # Add selected
        for k in self.laggedFeatures:
            key, lag = k.split('__lag__')
            self.input[k] = self.input[key].shift(-int(lag), fill_value=0)
        json.dump(self.laggedFeatures, open(self.folder + 'laggedFeatures_v%i.json' % self.version, 'w'))


    def select(self, inputFrame, outputFrame):
        # Check if not exists
        if os.path.exists(self.folder + 'FeatureSets_v%i.json' % self.version):
            return json.load(open(self.folder + 'FeatureSets_v%i.json' % self.version, 'r'))

        # Execute otherwise
        else:
            # Clean
            self._cleanAndSet(inputFrame, outputFrame)

            # Different Feature Sets
            pps = self._predictivePowerScore()
            rft, rfi = self._randomForestImportance()
            bp = self._borutaPy()
            return {'PPS': pps, 'RFT': rft, 'RFI': rfi, 'BP': bp}

    def _predictivePowerScore(self):
        '''
        Calculates the Predictive Power Score (https://github.com/8080labs/ppscore)
        Assymmetric correlation based on single decision trees trained on 5.000 samples with 4-Fold validation.
        '''
        print('[Features] Determining Features with PPS')
        data = self.input.copy()
        data['target'] = self.output.copy()
        pp_score = pps.predictors(data, "target")
        pp_cols = pp_score['x'][pp_score['ppscore'] != 0].to_list()
        print('[features] Selected %i features with Predictive Power Score' % len(pp_cols))
        return pp_cols

    def _randomForestImportance(self):
        '''
        Calculates Feature Importance with Random Forest, aka Mean Decrease in Gini Impurity
        Symmetric correlation based on multiple features and multiple trees ensemble
        '''
        print('[features] Determining Features with RF')
        if self.mode == 'regression':
            rf = RandomForestRegressor().fit(self.input, self.output)
        elif self.mode == 'classification':
            rf = RandomForestClassifier().fit(self.input, self.output)
        fi = rf.feature_importances_
        sfi = fi.sum()
        ind = np.flip(np.argsort(fi))
        # Info Threshold
        ind_keep = [ind[i] for i in range(len(ind)) if fi[ind[:i]].sum() <= self.informationThreshold * sfi]
        thresholded = self.input.keys()[ind_keep].to_list()
        ind_keep = [ind[i] for i in range(len(ind)) if fi[i] > sfi / 100]
        increment = self.input.keys()[ind_keep].to_list()
        print('[features] Selected %i features with RF thresholded' % len(thresholded))
        print('[features] Selected %i features with RF increment' % len(increment))
        return thresholded, increment

    def _borutaPy(self):
        print('[features] Determining Features with Boruta')
        if self.mode == 'regression':
            rf = RandomForestRegressor()
        elif self.mode == 'classification':
            rf = RandomForestClassifier()
        selector = BorutaPy(rf, n_estimators='auto', verbose=0)
        selector.fit(self.input.to_numpy(), self.output.to_numpy())
        bp_cols = self.input.keys()[selector.support_].to_list()
        print('[features] Selected %i features with Boruta' % len(bp_cols))
        return bp_cols

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
        self.data = data.astype('float32').fillna(0)
        self.output = output.astype('float32').fillna(0)

        # General settings
        self.seasonPeriods = seasonPeriods
        self.maxSamples = maxSamples        # Timeseries
        self.differ = differ                # Correlations
        self.lags = lags                    # Correlations

        # Storage settings
        self.tag = pretag
        self.folder = folder if folder == '' or folder[-1] == '/' else folder + '/'
        self.skip = skip_completed

        # Create dirs
        self.createDirs()
        self.run()


    def run(self):
        # Run all functions
        print('[EDA] Generating Missing Values Plot')
        self.missingValues()
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
            print('[EDA] Generating Scatter plots')
            self.scatters()
            print('[EDA] Generating SHAP plot')
            self.SHAP()
            print('[EDA] Generating Feature Ranking Plot')
            self.featureRanking()
            print('[EDA] Predictive Power Score Plot')
            self.predictivePowerScore()

    def createDirs(self):
        dirs = ['EDA', 'EDA/Boxplots', 'EDA/Seasonality', 'EDA/Colinearity', 'EDA/Lags', 'EDA/Correlation',
                'EDA/Correlation/ACF', 'EDA/Correlation/PACF', 'EDA/Correlation/Cross', 'EDA/NonLinear Correlation',
                'EDA/Timeplots', 'EDA/ScatterPlots/']
        for period in self.seasonPeriods:
            dirs.append(self.folder + 'EDA/Seasonality/' + str(period))
        for dir in dirs:
            try:
                os.makedirs(self.folder + dir)
                if dir == 'EDA/Correlation':
                    file = open(self.folder + 'EDA/Correlation/Readme.txt', 'w')
                    edit = file.write(
                        'Correlation Interpretation\n\nIf the series has positive autocorrelations for a high number of lags,\nit probably needs a higher order of differencing. If the lag-1 autocorrelation\nis zero or negative, or the autocorrelations are small and patternless, then \nthe series does not need a higher order of differencing. If the lag-1 autocorrleation\nis below -0.5, the series is likely to be overdifferenced. \nOptimum level of differencing is often where the variance is lowest. ')
                    file.close()
            except:
                continue

    def missingValues(self):
        if self.tag + 'MissingValues.png' in os.listdir(self.folder + 'EDA/'):
            return
        import missingno
        ax = missingno.matrix(self.data, figsize=[24, 16])
        fig = ax.get_figure()
        fig.savefig(self.folder + 'EDA/MissingValues.png')

    def boxplots(self):
        for key in tqdm(self.data.keys()):
            if self.tag + key + '.png' in os.listdir(self.folder + 'EDA/Boxplots/'):
                continue
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
        for key in tqdm(data.keys()):
            if self.tag + key + '.png' in os.listdir(self.folder + 'EDA/Timeplots/'):
                continue
            fig = plt.figure(figsize=[24, 16])
            plt.plot(data.index, data[key])
            plt.title(key)
            fig.savefig(self.folder + 'EDA/Timeplots/' + self.tag + key + '.png', format='png', dpi=100)
            plt.close(fig)

    def seasonality(self):
        for key in tqdm(self.data.keys()):
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
            return
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
            return
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
        for key in tqdm(self.data.keys()):
            if self.tag + key + '_differ_' + str(self.differ) + '.png' in os.listdir(self.folder + 'EDA/Correlation/ACF/'):
                continue
            fig = plot_acf(self.data[key], fft=True)
            plt.title(key)
            fig.savefig(self.folder + 'EDA/Correlation/ACF/' + self.tag + key + '_differ_' + str(self.differ) + '.png', format='png', dpi=300)
            plt.close()

    def partialAutoCorr(self):
        for key in tqdm(self.data.keys()):
            if self.tag + key + '_differ_' + str(self.differ) + '.png' in os.listdir(self.folder + 'EDA/Correlation/PACF/'):
                continue
            try:
                fig = plot_pacf(self.data[key])
                fig.savefig(self.folder + 'EDA/Correlation/PACF/' + self.tag + key + '_differ_' + str(self.differ) + '.png', format='png', dpi=300)
                plt.title(key)
                plt.close()
            except:
                continue

    def crossCorr(self):
        folder = 'EDA/Correlation/Cross/'
        output = self.output.to_numpy().reshape((-1))
        for key in tqdm(self.data.keys()):
            if self.tag + key + '_differ_' + str(self.differ) + '.png' in os.listdir(self.folder + folder):
                continue
            try:
                fig = plt.figure(figsize=[24, 16])
                plt.xcorr(self.data[key], output, maxlags=self.lags)
                plt.title(key)
                fig.savefig(self.folder + folder + self.tag + key + '_differ_' + str(self.differ) + '.png', format='png', dpi=300)
                plt.close()
            except:
                continue

    def scatters(self):
        # Plots
        for key in tqdm(self.data.keys()):
            if self.tag + key + '.png' in os.listdir(self.folder + 'EDA/ScatterPlots/'):
                continue
            fig = plt.figure(figsize=[24, 16])
            plt.scatter(self.output, self.data[key], alpha=0.2)
            plt.xlabel('Output')
            plt.ylabel(key)
            plt.title('Scatterplot ' + key + ' - output')
            fig.savefig(self.folder + 'EDA/ScatterPlots/' + self.tag + key + '.png', format='png', dpi=100)
            plt.close(fig)

    def SHAP(self, args={}):
        if self.tag + 'SHAP.png' in os.listdir(self.folder + 'EDA/Nonlinear Correlation'):
            return
        if set(self.output) == {1, -1}:
            model = RandomForestClassifier(**args).fit(self.data, self.output)
        else:
            model = RandomForestRegressor(**args).fit(self.data, self.output)

        import shap
        fig = plt.figure(figsize=[8, 32])
        plt.subplots_adjust(left=0.4)
        shap_values = shap.TreeExplainer(model).shap_values(self.data)
        shap.summary_plot(shap_values, self.data, plot_type='bar')
        fig.savefig(self.folder + 'EDA/Nonlinear Correlation/' + self.tag + 'SHAP.png', format='png', dpi=300)

    def featureRanking(self, args={}):
        if self.tag + 'RF.png' in os.listdir(self.folder + 'EDA/Nonlinear Correlation'):
            return
        if set(self.output) == {1, -1}:
            model = RandomForestClassifier(**args).fit(self.data, self.output)
        else:
            model = RandomForestRegressor(**args).fit(self.data, self.output)
        fig, ax = plt.subplots(figsize=[8, 12], constrained_layout=True)
        ind = np.argsort(model.feature_importances_)
        plt.barh(list(self.data.keys()[ind])[-15:], width=model.feature_importances_[ind][-15:])
        results = pd.DataFrame({'x': self.data.keys(), 'score': model.feature_importances_})
        results.to_csv(self.folder + 'EDA/Nonlinear Correlation/' + self.tag + 'RF.csv')
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
        pp_score = pps.predictors(data, 'target').sort_values('ppscore')
        fig, ax = plt.subplots(figsize=[8, 12], constrained_layout=True)
        plt.barh(pp_score['x'][-15:], width=pp_score['ppscore'][-15:])
        fig.savefig(self.folder + 'EDA/Nonlinear Correlation/' + self.tag + 'Ppscore.png', format='png', dpi=400)
        pp_score.to_csv(self.folder + 'EDA/Nonlinear Correlation/pp_score.csv')
        plt.close()

class Modelling(object):

    def __init__(self, mode='regression', shuffle=False, plot=False,
                 folder='models/', n_splits=3, dataset=0, store_models=False, store_results=True):
        self.mode = mode
        self.shuffle = shuffle
        self.plot = plot
        self.acc = []
        self.std = []
        self.nSplits = n_splits
        self.dataset = str(dataset)
        self.store_results = store_results
        self.store_models = store_models
        self.folder = folder if folder[-1] == '/' else folder + '/'

    def fit(self, input, output):
        if self.mode == 'regression':
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.nSplits, shuffle=self.shuffle)
            return self._fit(input, output, cv, Metrics.mae)
        if self.mode == 'classification':
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=self.nSplits, shuffle=self.shuffle)
            return self._fit(input, output, cv, Metrics.acc)

    def return_models(self):
        from sklearn import linear_model, svm, neighbors, tree, ensemble, neural_network
        from sklearn.experimental import enable_hist_gradient_boosting
        import catboost
        if self.mode == 'classification':
            ridge = linear_model.RidgeClassifier()
            lasso = linear_model.Lasso()
            sgd = linear_model.SGDClassifier()
            svc = svm.SVC(kernel='rbf')
            knc = neighbors.KNeighborsClassifier()
            dtc = tree.DecisionTreeClassifier()
            ada = ensemble.AdaBoostClassifier()
            cat = catboost.CatBoostClassifier(verbose=0)
            bag = ensemble.BaggingClassifier()
            gbc = ensemble.GradientBoostingClassifier()
            hgbc = ensemble.HistGradientBoostingClassifier()
            rfc = ensemble.RandomForestClassifier()
            mlp = neural_network.MLPClassifier()
            return [ridge, lasso, sgd, svc, knc, dtc, ada, cat, bag, gbc, hgbc, mlp]
        elif self.mode == 'regression':
            ridge = linear_model.Ridge()
            lasso = linear_model.Lasso()
            sgd = linear_model.SGDRegressor()
            svr = svm.SVR(kernel='rbf')
            knr = neighbors.KNeighborsRegressor()
            dtr = tree.DecisionTreeRegressor()
            ada = ensemble.AdaBoostRegressor()
            cat = catboost.CatBoostRegressor(verbose=0)
            gbr = ensemble.GradientBoostingRegressor()
            hgbr = ensemble.HistGradientBoostingRegressor()
            rfr = ensemble.RandomForestRegressor()
            mlp = neural_network.MLPRegressor()
            return [ridge, lasso, sgd, svr, knr, dtr, ada, cat, gbr, hgbr, mlp]

    def _fit(self, input, output, cross_val, metric):
        # Convert to NumPy
        input = np.array(input)
        output = np.array(output)

        # Data
        print('[modelling] Splitting data (shuffle=%s, splits=%i, features=%i)' % (str(self.shuffle), self.nSplits, len(input[0])))

        if self.store_results and 'Initial_Models.csv' in os.listdir(self.folder):
            results = pd.read_csv(self.folder + 'Initial_Models.csv')
        else:
            results = pd.DataFrame(columns=['date', 'model', 'dataset', 'params', 'mean_score', 'std_score', 'mean_time', 'std_time'])

        # Models
        self.models = self.return_models()

        # Loop through models
        for master_model in self.models:
            # Check first if we don't already have the results
            ind = np.where(np.logical_and(np.logical_and(
                results['model'] == type(master_model).__name__,
                results['dataset'] == self.dataset),
                results['date'] == datetime.today().strftime('%d %b %Y, %Hh')))[0]
            if len(ind) != 0:
                self.acc.append(results.iloc[ind[0]]['mean_score'])
                self.std.append(results.iloc[ind[0]]['std_score'])
                continue

            # Time & loops through Cross-Validation
            v_acc = []
            t_acc = []
            t_train = []
            for t, v in cross_val.split(input, output):
                t_start = time.time()
                ti, vi, to, vo = input[t], input[v], output[t], output[v]
                model = copy.copy(master_model)
                model.fit(ti, to)
                v_acc.append(metric(vo, model.predict(vi)))
                t_acc.append(metric(to, model.predict(ti)))
                t_train.append(time.time() - t_start)

            # Results
            results = results.append({'date': datetime.today().strftime('%d %b %Y'), 'model': type(model).__name__,
                                      'dataset': self.dataset, 'params': model.get_params(),
                                      'mean_score': np.mean(v_acc),
                                      'std_score': np.std(v_acc), 'mean_time': np.mean(t_train),
                                      'std_time': np.std(t_train)}, ignore_index=True)
            self.acc.append(np.mean(v_acc))
            self.std.append(np.std(v_acc))
            if self.store_models:
                joblib.dump(model, self.folder + type(model).__name__ + '_%.5f.joblib' % self.acc[-1])
            if self.plot:
                plt.plot(p, alpha=0.2)
            print('[modelling] %s train/val: %.2f %.2f, training time: %.1f s' %
                  (type(model).__name__.ljust(60), np.mean(t_acc), np.mean(v_acc), time.time() - t_start))

        # Store CSV
        if self.store_results:
            results.to_csv(self.folder + 'Initial_Models.csv')

        # Plot
        if self.plot:
            plt.plot(vo, c='k')
            if self.regression:
                ind = np.where(self.acc == np.min(self.acc))[0][0]
            else:
                ind = np.where(self.acc == np.max(self.acc))[0][0]
            p = self.models[ind].predict(vi)
            plt.plot(p, color='#2369ec')
            plt.title('Predictions')
            plt.legend(['True output', 'Ridge', 'Lasso', 'SGD', 'KNR', 'DTR', 'ADA', 'GBR', 'HGBR', 'MLP', 'Best'])
            plt.ylabel('Output')
            plt.xlabel('Samples')
            plt.show()
        return results

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
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.core.series.Series):
            assert isinstance(output, pd.DataFrame) or isinstance(output, pd.core.series.Series), \
                'Input and Output need to be the same data type.'
            return self.convert_pandas(input, output)
        elif isinstance(input, np.ndarray):
            assert isinstance(output, np.ndarray), 'Input and Output need to be the same data type.'
            return self.convert_numpy(input, output, flat=flat)
        else:
            TypeError('Input & Output need to be same datatype, either Numpy or Pandas.')

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
        for i, param in tqdm(enumerate(self.parsed_params)):
            # print('[GridSearch] ', param)
            scoring = []
            timing = []
            for train_ind, val_ind in self.cv.split(input, output):
                # Start Timer
                t = time.time()

                # Split data
                xtrain, xval = input[train_ind], input[val_ind]
                ytrain, yval = output[train_ind], output[val_ind]

                # Model training
                model = copy.copy(self.model)
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

            # print('[GridSearch] [%s] Score: %.4f \u00B1 %.4f (in %.1f seconds) (Best score so-far: %.4f \u00B1 %.4f) (%i / %i)' %
            #       (datetime.now().strftime('%H:%M'), np.mean(scoring), np.std(scoring), np.mean(timing), self.best[0], self.best[1], i + 1, len(self.parsed_params)))
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
