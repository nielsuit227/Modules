import copy
import functools
import inspect
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import textwrap
import time
import warnings
from datetime import datetime
from json import JSONDecodeError

import catboost
import joblib
import lightgbm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns
import sklearn
import xgboost
from boruta import BorutaPy
from scipy.stats import randint, uniform
from sklearn import cluster, metrics, tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')
# warnings.filterwarnings(action='ignore', category=UserWarning)
# pd.options.mode.chained_assignment = None

# Priority
# implement warning for imputing keys
# print initial modelling when already completed
# Parameterize _getHyperParameters()
# Change resources of halving grid search to samples when samples > 25.000
# Stacking Builder
# For metrics per class, make sure that all is outputted by Modelling and Pipeline.validatecal

# Nicety
# add report output
# implement ensembles (http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
# implement autoencoder for Feature Extraction
# implement transformers


def printTime(func):
    def wrap(*args, **kwargs):
        t = time.time()
        x = func(*args, **kwargs)
        print(func.__name__, time.time() - t)
        return x

    return wrap


class Utils:
    @staticmethod
    def booleanInput(question):
        x = input(question + " [y / n]")
        if x.lower() == "n" or x.lower() == "no":
            return False
        elif x.lower() == "y" or x.lower() == "yes":
            return True
        else:
            print('Sorry, I did not understand. Please answer with "n" or "y"')
            return self.booleanInput(question)

    @staticmethod
    def notification(notification):
        import requests

        url = "https://slack.com/api/chat.postMessage"
        data = {
            "token": os.getenv("SLACK_OAUTH_TOKEN"),
            "channel": "automl",
            "text": message,
            "username": "AutoML",
        }
        requests.post(url, data=data)

    @staticmethod
    def parseJson(json_string):
        if isinstance(json_string, dict):
            return json_string
        else:
            try:
                return json.loads(
                    json_string.replace("'", '"')
                    .replace("True", "true")
                    .replace("False", "false")
                    .replace("nan", "NaN")
                    .replace("None", "null")
                )
            except JSONDecodError:
                print("[AutoML] Cannot validate, imparsable JSON.")
                print(json_string)
                return json_string

    def crossFeaturesMP(self, data, labels):
        processes = []
        print("[Features] Initializing all cross-features")
        time.sleep(0.02)
        for i in tqdm(range(data.shape[1])):
            for j in range(data.shape[1]):
                if i < j:
                    continue
                processes.append(((i, j), "{}x{}".format(i, j)))
        print("[Features] Analysing all cross-features")
        time.sleep(0.02)
        print(processes[0])
        scores = process_map(
            functools.partial(Utils.crossFeature, data, labels),
            processes,
            max_workers=4,
            chunksize=min(25, int(len(processes) / 8 / 8)),
        )
        # with mp.Pool() as p:
        #     scores = p.starmap(functools.partial(Utils.crossFeature, data, labels),
        #                   iter(processes))
        return scores

    @staticmethod
    def crossFeature(data, labels, arg):
        feature, name = arg
        feature = data[:, feature[0]] * data[:, feature[1]]
        feature = feature.reshape((-1, 1))
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(feature, labels)
        return (name, model.score(feature, labels))


class Pipeline(object):
    def __init__(
        self,
        target,
        project="",
        version=None,
        mode="regression",
        objective=None,
        fast_run=False,
        # Data Processing
        num_cols=[],
        date_cols=[],
        cat_cols=[],
        missing_values="interpolate",
        outlier_removal="clip",
        z_score_threshold=4,
        include_output=False,
        # Feature Processing
        max_lags=10,
        max_diff=2,
        information_threshold=0.99,
        extract_features=True,
        # Sequencing
        sequence=False,
        back=0,
        forward=0,
        shift=0,
        diff="none",
        # Initial Modelling
        normalize=True,
        shuffle=False,
        cv_splits=3,
        store_models=True,
        # Grid Search
        grid_search_iterations=3,
        use_halving=True,
        # Production
        custom_code="",
        # Flags
        plot_eda=None,
        process_data=None,
        validate_result=None,
        verbose=1,
    ):
        # Starting
        print("\n\n*** Starting Amplo AutoML - %s ***\n\n" % project)

        # Parsing input
        if len(project) == 0:
            self.mainDir = "AutoML/"
        else:
            self.mainDir = project if project[-1] == "/" else project + "/"
        self.target = re.sub("[^a-z0-9]", "_", target.lower())
        self.verbose = verbose
        self.customCode = custom_code
        self.fastRun = fast_run

        # Checks
        assert (
            mode == "regression" or mode == "classification"
        ), "Supported modes: regression, classification."
        assert isinstance(
            target, str
        ), "Target needs to be of type string, key of target"
        assert isinstance(project, str), "Project is a name, needs to be of type string"
        assert isinstance(num_cols, list), "Num cols must be a list of strings"
        assert isinstance(date_cols, list), "Date cols must be a list of strings"
        assert isinstance(cat_cols, list), "Cat Cols must be a list of strings"
        assert isinstance(shift, int), "Shift needs to be an integer"
        assert isinstance(max_diff, int), "max_diff needs to be an integer"
        assert max_lags < 50, "Max_lags too big. Max 50."
        assert (
            information_threshold > 0 and information_threshold < 1
        ), "Information threshold needs to be within [0, 1"
        assert max_diff < 5, "Max difftoo big. Max 5."

        # Objective
        if objective is not None:
            self.objective = objective
        else:
            if mode == "regression:":
                self.objective = "neg_mean_squared_error"
            elif mode == "classification":
                self.objective = "accuracy"
        assert isinstance(objective, str), "Objective needs to be a string."
        assert (
            objective in metrics.SCORERS.keys()
        ), "Metric not supported, look at sklearn.metrics.SCORERS.keys()"
        self.scorer = metrics.SCORERS[self.objective]

        # Params needed
        self.mode = mode
        self.version = version
        self.numCols = num_cols
        self.dateCols = date_cols
        self.catCols = cat_cols
        self.missingValues = missing_values
        self.outlierRemoval = outlier_removal
        self.zScoreThreshold = z_score_threshold
        self.includeOutput = include_output
        self.sequence = sequence
        self.sequenceBack = back
        self.sequenceForward = forward
        self.sequenceShift = shift
        self.sequenceDiff = diff
        self.normalize = normalize
        self.shuffle = shuffle
        self.cvSplits = cv_splits
        self.gridSearchIterations = grid_search_iterations
        self.useHalvingGridSearch = use_halving
        self.plotEDA = plot_eda
        self.processData = process_data
        self.validateResults = validate_result

        # Instance initiating
        self.X = None
        self.Y = None
        self.colKeep = None
        self.results = None

        # Flags
        self._setFlags()

        # Create Directories
        self._createDirs()

        # Load Version
        self._loadVersion()

        # Sub- Classes
        self.DataProcessing = DataProcessing(
            target=self.target,
            num_cols=num_cols,
            date_cols=date_cols,
            cat_cols=cat_cols,
            missing_values=missing_values,
            mode=mode,
            outlier_removal=outlier_removal,
            z_score_threshold=z_score_threshold,
            folder=self.mainDir + "Data/",
            version=self.version,
        )
        self.FeatureProcessing = FeatureProcessing(
            max_lags=max_lags,
            max_diff=max_diff,
            extract_features=extract_features,
            mode=mode,
            information_threshold=information_threshold,
            folder=self.mainDir + "Features/",
            version=self.version,
        )
        self.Sequence = Sequence(back=back, forward=forward, shift=shift, diff=diff)
        self.Modelling = Modelling(
            mode=mode,
            shuffle=shuffle,
            store_models=store_models,
            scoring=metrics.SCORERS[self.objective],
            store_results=False,
            folder=self.mainDir + "Models/",
        )

        # Store production
        self.bestModel = None
        self.bestFeatures = None
        self.bestScaler = None
        self.bestOScaler = None

    def _setFlags(self):
        if self.plotEDA is None:
            self.plotEDA = Utils.booleanInput("Make all EDA graphs?")
        if self.processData is None:
            self.processData = Utils.booleanInput("Process/prepare data?")
        if self.validateResults is None:
            self.validateResults = Utils.booleanInput("Validate results?")

    def _loadVersion(self):
        if self.version == None:
            versions = os.listdir(self.mainDir + "Production")
            # Updates changelog
            if self.processData:
                if len(versions) == 0:
                    if self.verbose > 0:
                        print("[AutoML] No Production files found. Setting version 0.")
                    self.version = 0
                    file = open(self.mainDir + "changelog.txt", "w")
                    file.write("Dataset changelog. \nv0: Initial")
                    file.close()
                else:
                    self.version = len((versions))

                    # Check if not already started
                    with open(self.mainDir + "changelog.txt", "r") as f:
                        changelog = f.read()

                    # Else ask for changelog
                    if "v%i" % self.version in changelog:
                        changelog = changelog[changelog.find("v%i" % self.version) :]
                        changelog = changelog[: max(0, changelog.find("\n"))]
                    else:
                        changelog = "\nv%i: " % self.version + input(
                            "Data changelog v%i:\n" % self.version
                        )
                        file = open(self.mainDir + "changelog.txt", "a")
                        file.write(changelog)
                        file.close()
                    if self.verbose > 0:
                        print("[AutoML] Set version %s" % (changelog[1:]))
            else:
                if len(versions) == 0:
                    if self.verbose > 0:
                        print("[AutoML] No Production files found. Setting version 0.")
                    self.version = 0
                else:
                    self.version = int(len(versions)) - 1
                    with open(self.mainDir + "changelog.txt", "r") as f:
                        changelog = f.read()
                    changelog = changelog[changelog.find("v%i" % self.version) :]
                    if self.verbose > 0:
                        print(
                            "[AutoML] Loading last version (%s)"
                            % changelog[: changelog.find("\n")]
                        )

    def _createDirs(self):
        dirs = ["", "Data", "Features", "Models", "Production", "Validation", "Sets"]
        for dir in dirs:
            try:
                os.makedirs(self.mainDir + dir)
            except:
                continue

    def _sortResults(self, results):
        results["worst_case"] = results["mean_objective"] - results["std_objective"]
        return results.sort_values("worst_case", ascending=False)

    def _getBestParams(self, model, feature_set):
        # Filter results for model and version
        results = self.results[
            np.logical_and(
                self.results["model"] == type(model).__name__,
                self.results["data_version"] == self.version,
            )
        ]

        # Filter results for feature set & sort them
        results = self._sortResults(results[results["dataset"] == feature_set])

        # Warning for unoptimized results
        if "Hyperparameter Opt" not in results["type"].values:
            warnings.warn("Hyperparameters not optimized for this combination")

        # Parse & return best parameters (regardless of if it's optimized)
        return Utils.parseJson(results.iloc[0]["params"])

    def fit(self, data):
        """
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
        """
        # Execute pipeline
        self._dataProcessing(data)
        self._eda(data)
        self._featureProcessing()
        self._initialModelling()
        self.gridSearch()
        # Production Env
        if (
            not os.path.exists(self.mainDir + "Production/v%i/" % self.version)
            or len(os.listdir(self.mainDir + "Production/v%i/" % self.version)) == 0
        ):
            self._prepareProductionFiles()
        print("[AutoML] Done :)")

    def _eda(self, data):
        if self.plotEDA:
            print("[AutoML] Starting Exploratory Data Analysis")
            self.eda = ExploratoryDataAnalysis(
                self.X, Y=self.Y, folder=self.mainDir, version=self.version
            )

    def _dataProcessing(self, data):
        # Load if possible
        if os.path.exists(self.mainDir + "Data/Cleaned_v%i.csv" % self.version):
            print("[AutoML] Loading Cleaned Data")
            data = pd.read_csv(
                self.mainDir + "Data/Cleaned_v%i.csv" % self.version, index_col="index"
            )

        # Clean
        else:
            print("[AutoML] Cleaning Data")
            data = self.DataProcessing.clean(data)

        # Split and store in memory
        self.Y = data[[self.target]]
        self.X = data
        if self.includeOutput is False:
            self.X = self.X.drop(self.target, axis=1)

        # Assert classes in case of classification
        if self.mode == "classification":
            if self.Y[self.target].nunique() >= 50:
                warnings.warn("More than 50 classes, you might want to reconsider")
            if set(self.Y[self.target]) != set(
                [i for i in range(len(set(self.Y[self.target])))]
            ):
                warnings.warn("Classes should be [0, 1, ...]")

    def _featureProcessing(self):
        # Extract
        self.X = self.FeatureProcessing.extract(self.X, self.Y[self.target])

        # Select
        self.colKeep = self.FeatureProcessing.select(self.X, self.Y[self.target])

    def _initialModelling(self):
        # Load existing results
        if "Results.csv" in os.listdir(self.mainDir):
            self.results = pd.read_csv(self.mainDir + "Results.csv")
            self.Modelling.samples = len(self.Y)

        # Check if this version has been modelled
        if (
            self.results is not None
            and (self.version == self.results["data_version"]).any()
        ):
            self.results = self._sortResults(self.results)

        # Run Modelling
        else:
            init = []
            for set, cols in self.colKeep.items():
                # Skip empty sets
                if len(cols) == 0:
                    print("[AutoML] Skipping %s features, empty set" % set)
                else:
                    print(
                        "[AutoML] Initial Modelling for %s features (%i)"
                        % (set, len(cols))
                    )

                    # Apply Feature Set
                    self.Modelling.dataset = set
                    # X, Y = self.X.reindex(columns=cols), self.Y.loc[:, self.target]
                    X, Y = self.X[cols], self.Y

                    # Normalize Feature Set (Done here to get one normalization file per feature set)
                    if self.normalize:
                        normalizeFeatures = [
                            k for k in X.keys() if k not in self.dateCols + self.catCols
                        ]
                        scaler = StandardScaler()
                        X[normalizeFeatures] = scaler.fit_transform(
                            X[normalizeFeatures]
                        )
                        pickle.dump(
                            scaler,
                            open(
                                self.mainDir
                                + "Features/Scaler_%s_%i.pickle" % (set, self.version),
                                "wb",
                            ),
                        )
                        if self.mode == "regression":
                            oScaler = StandardScaler()
                            Y[self.target] = oScaler.fit_transform(Y)
                            pickle.dump(
                                oScaler,
                                open(
                                    self.mainDir
                                    + "Features/OScaler_%s_%i.pickle"
                                    % (set, self.version),
                                    "wb",
                                ),
                            )

                    # Sequence if necessary
                    if self.sequence:
                        X, Y = self.Sequence.convert(X, Y)

                    # Do the modelling
                    results = self.Modelling.fit(X, Y)

                    # Add results to memory
                    results["type"] = "Initial modelling"
                    results["data_version"] = self.version
                    if self.results is None:
                        self.results = results
                    else:
                        self.results = self.results.append(results)

            # Save results
            self.results.to_csv(self.mainDir + "Results.csv", index=False)

    def _getHyperParams(self, model):
        # Parameters for both Regression / Classification
        if (
            isinstance(model, sklearn.linear_model.Lasso)
            or isinstance(model, sklearn.linear_model.Ridge)
            or isinstance(model, sklearn.linear_model.RidgeClassifier)
        ):
            return {
                "alpha": uniform(0, 10),
            }
        elif isinstance(model, sklearn.svm.SVC) or isinstance(model, sklearn.svm.SVR):
            return {
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 0.5, 1],
                "C": uniform(0, 10),
            }
        elif isinstance(model, sklearn.neighbors.KNeighborsRegressor) or isinstance(
            model, sklearn.neighbors.KNeighborsClassifier
        ):
            return {
                "n_neighbors": randint(5, 50),
                "weights": ["uniform", "distance"],
                "leaf_size": randint(10, 150),
                "n_jobs": [mp.cpu_count() - 1],
            }
        elif isinstance(
            model, sklearn.neural_network._multilayer_perceptron.MLPClassifier
        ) or isinstance(
            model, sklearn.neural_network._multilayer_perceptron.MLPRegressor
        ):
            return {
                "hidden_layer_sizes": [
                    (100,),
                    (100, 100),
                    (100, 50),
                    (200, 200),
                    (200, 100),
                    (200, 50),
                    (50, 50, 50, 50),
                ],
                "learning_rate": ["adaptive", "invscaling"],
                "alpha": [1e-4, 1e-3, 1e-5],
                "shuffle": [False],
            }

        # Regressor specific hyperparameters
        elif self.mode == "regression":
            if isinstance(model, sklearn.linear_model.SGDRegressor):
                return {
                    "loss": [
                        "squared_loss",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "alpha": randint(0, 5),
                }
            elif isinstance(model, sklearn.tree.DecisionTreeRegressor):
                return {
                    "criterion": ["mse", "friedman_mse", "mae", "poisson"],
                    "max_depth": randint(5, 50),
                }
            elif isinstance(model, sklearn.ensemble.AdaBoostRegressor):
                return {
                    "n_estimators": randint(25, 250),
                    "loss": ["linear", "square", "exponential"],
                    "learning_rate": uniform(0, 1),
                }
            elif isinstance(model, catboost.core.CatBoostRegressor):
                return {
                    "loss_function": ["MAE", "RMSE"],
                    "learning_rate": uniform(0, 1),
                    "l2_leaf_reg": uniform(0, 10),
                    "depth": randint(3, 15),
                    "min_data_in_leaf": randint(1, 1000),
                    "max_leaves": randint(10, 250),
                }
            elif isinstance(model, sklearn.ensemble.GradientBoostingRegressor):
                return {
                    "loss": ["ls", "lad", "huber"],
                    "learning_rate": uniform(0, 1),
                    "max_depth": randint(3, 15),
                }
            elif isinstance(model, sklearn.ensemble.HistGradientBoostingRegressor):
                return {
                    "max_iter": randint(100, 250),
                    "max_bins": randint(100, 255),
                    "loss": ["least_squares", "least_absolute_deviation"],
                    "l2_regularization": uniform(0, 10),
                    "learning_rate": uniform(0, 1),
                    "max_leaf_nodes": randint(30, 150),
                    "early_stopping": [True],
                }
            elif isinstance(model, sklearn.ensemble.RandomForestRegressor):
                return {
                    "criterion": ["mse", "mae"],
                    "max_depth": randint(3, 15),
                    "max_features": ["auto", "sqrt"],
                    "min_samples_split": randint(2, 50),
                    "min_samples_leaf": randint(1, 1000),
                    "bootstrap": [True, False],
                }
            elif model.__module__ == "xgboost.sklearn":
                return {
                    "max_depth": randint(3, 15),
                    "booster": ["gbtree", "gblinear", "dart"],
                    "learning_rate": uniform(0, 10),
                    "verbosity": [0],
                    "n_jobs": [mp.cpu_count() - 1],
                }
            elif model.__module__ == "lightgbm.sklearn":
                return {
                    "num_leaves": randint(10, 150),
                    "min_child_samples": randint(1, 1000),
                    "min_child_weight": uniform(0, 1),
                    "subsample": uniform(0, 1),
                    "colsample_bytree": uniform(0, 1),
                    "reg_alpha": uniform(0, 1),
                    "reg_lambda": uniform(0, 1),
                    "n_jobs": [mp.cpu_count() - 1],
                }

        # Classification specific hyperparameters
        elif self.mode == "classification":
            if isinstance(model, sklearn.linear_model.SGDClassifier):
                return {
                    "loss": ["hinge", "log", "modified_huber", "squared_hinge"],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "alpha": uniform(0, 10),
                    "max_iter": randint(250, 2000),
                }
            elif isinstance(model, sklearn.tree.DecisionTreeClassifier):
                return {
                    "criterion": ["gini", "entropy"],
                    "max_depth": randint(5, 50),
                }
            elif isinstance(model, sklearn.ensemble.AdaBoostClassifier):
                return {
                    "n_estimators": randint(25, 250),
                    "learning_rate": uniform(0, 1),
                }
            elif isinstance(model, catboost.core.CatBoostClassifier):
                return {
                    "loss_function": [
                        "Logloss"
                        if self.Y[self.target].nunique() == 2
                        else "MultiClass"
                    ],
                    "learning_rate": uniform(0, 1),
                    "l2_leaf_reg": uniform(0, 10),
                    "depth": randint(1, 10),
                    "min_data_in_leaf": randint(50, 500),
                    "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
                }
            elif isinstance(model, sklearn.ensemble.BaggingClassifier):
                return {
                    # 'n_estimators': [5, 10, 15, 25, 50],
                    "max_features": uniform(0, 1),
                    "bootstrap": [False, True],
                    "bootstrap_features": [True, False],
                    "n_jobs": [mp.cpu_count() - 1],
                }
            elif isinstance(model, sklearn.ensemble.GradientBoostingClassifier):
                return {
                    "loss": ["deviance", "exponential"],
                    "learning_rate": uniform(0, 1),
                    "max_depth": randint(3, 15),
                }
            elif isinstance(model, sklearn.ensemble.HistGradientBoostingClassifier):
                return {
                    "max_iter": randint(100, 250),
                    "max_bins": randint(100, 255),
                    "l2_regularization": uniform(0, 10),
                    "learning_rate": uniform(0, 1),
                    "max_leaf_nodes": randint(30, 150),
                    "early_stopping": [True],
                }
            elif isinstance(model, sklearn.ensemble.RandomForestClassifier):
                return {
                    "criterion": ["gini", "entropy"],
                    "max_depth": randint(3, 15),
                    "max_features": ["auto", "sqrt"],
                    "min_samples_split": randint(2, 50),
                    "min_samples_leaf": randint(1, 1000),
                    "bootstrap": [True, False],
                }
            elif model.__module__ == "xgboost.sklearn":
                return {
                    "max_depth": randint(3, 15),
                    "booster": ["gbtree", "gblinear", "dart"],
                    "learning_rate": uniform(0, 10),
                    "verbosity": [0],
                    "n_jobs": [mp.cpu_count() - 1],
                    "scale_pos_weight": uniform(0, 100),
                }
            elif model.__module__ == "lightgbm.sklearn":
                return {
                    "num_leaves": randint(10, 150),
                    "min_child_samples": randint(1, 1000),
                    "min_child_weight": uniform(0, 1),
                    "subsample": uniform(0, 1),
                    "colsample_bytree": uniform(0, 1),
                    "reg_alpha": uniform(0, 1),
                    "reg_lambda": uniform(0, 1),
                    "n_jobs": [mp.cpu_count() - 1],
                }

        # Raise error if nothing is returned
        raise NotImplementedError(
            "Hyperparameter tuning not implemented for ", type(model).__name__
        )

    def gridSearch(self, model=None, feature_set=None, params=None):
        """
        Runs a grid search. By default, takes the self.results, and runs for the top 3 optimizations.
        There is the option to provide a model & feature_set, but both have to be provided. In this case,
        the model & data set combination will be optimized.
        :param model:
        :param params:
        :param feature_set:
        :return:
        """
        assert (
            model is not None and feature_set is not None or model == feature_set
        ), "Model & feature_set need to be either both None or both provided."

        # If arguments are provided
        if model is not None:

            # Get model string
            if isinstance(model, str):
                models = self.Modelling.return_models()
                model = models[
                    [
                        i
                        for i in range(len(models))
                        if type(models[i]).__name__ == model
                    ][0]
                ]

            # Organise existing results
            results = self.results[
                np.logical_and(
                    self.results["model"] == type(model).__name__,
                    self.results["data_version"] == self.version,
                )
            ]
            results = self._sortResults(results[results["dataset"] == feature_set])

            # Check if exists and load
            if ("Hyperparameter Opt" == results["type"]).any():
                print("[AutoML] Loading optimization results.")
                hyperOptResults = results[results["type"] == "Hyperparameter Opt"]
                params = Utils.parseJson(hyperOptResults.iloc[0]["params"])

            # Or run
            else:
                # Parameter check
                if params is None:
                    params = self._getHyperParams(model)

                # Run Grid Search
                if self.useHalvingGridSearch:
                    gridSearchResults = self._sortResults(
                        self._gridSearchIterationHalvingCV(model, params, feature_set)
                    )
                else:
                    gridSearchResults = self._sortResults(
                        self._gridSearchIteration(model, params, feature_set)
                    )

                # Store results
                gridSearchResults["model"] = type(model).__name__
                gridSearchResults["data_version"] = self.version
                gridSearchResults["dataset"] = feature_set
                gridSearchResults["type"] = "Hyperparameter Opt"
                self.results = self.results.append(gridSearchResults)
                self.results.to_csv(self.mainDir + "Results.csv", index=False)

                # Get params for validation
                params = Utils.parseJson(results.iloc[0]["params"])

            # Validate
            self._validateResult(model, params, feature_set)
            return

        # If arguments aren't provided
        models = self.Modelling.return_models()
        results = self._sortResults(
            self.results[
                np.logical_and(
                    self.results["type"] == "Initial modelling",
                    self.results["data_version"] == self.version,
                )
            ]
        )

        # TODO Check if model is not already opitmized for a feature set
        completedModels = list(
            set(
                self.results[
                    np.logical_and(
                        self.results["type"] == "Hyperparameter Opt",
                        self.results["data_version"] == self.version,
                    )
                ]
            )
        )
        for iteration in range(self.gridSearchIterations):
            # Grab settings
            settings = results.iloc[iteration]
            model = models[
                [
                    i
                    for i in range(len(models))
                    if type(models[i]).__name__ == settings["model"]
                ][0]
            ]
            featureSet = settings["dataset"]

            # Check whether exists
            modelResults = self.results[
                np.logical_and(
                    self.results["model"] == type(model).__name__,
                    self.results["data_version"] == self.version,
                )
            ]
            modelResults = self._sortResults(
                modelResults[modelResults["dataset"] == featureSet]
            )

            # If exists
            if ("Hyperparameter Opt" == modelResults["type"]).any():
                hyperOptRes = modelResults[modelResults["type"] == "Hyperparameter Opt"]
                params = Utils.parseJson(hyperOptRes.iloc[0]["params"])

            # Else run
            else:
                params = self._getHyperParams(model)
                if self.useHalvingGridSearch:
                    gridSearchResults = self._sortResults(
                        self._gridSearchIterationHalvingCV(model, params, featureSet)
                    )
                else:
                    gridSearchResults = self._sortResults(
                        self._gridSearchIteration(model, params, featureSet)
                    )

                # Store
                gridSearchResults["model"] = type(model).__name__
                gridSearchResults["data_version"] = self.version
                gridSearchResults["dataset"] = featureSet
                gridSearchResults["type"] = "Hyperparameter Opt"
                self.results = self.results.append(gridSearchResults)
                self.results.to_csv(self.mainDir + "Results.csv", index=False)
                params = Utils.parseJson(gridSearchResults.iloc[0]["params"])

            # Validate
            if self.validateResults:
                self._validateResult(model, params, featureSet)

    def _gridSearchIteration(self, model, params, feature_set):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        print(
            "\n[AutoML] Starting Hyperparameter Optimization for %s on %s features (%i samples, %i features)"
            % (
                type(model).__name__,
                feature_set,
                len(self.X),
                len(self.colKeep[feature_set]),
            )
        )

        # Select data
        X, Y = self.X[self.colKeep[feature_set]], self.Y

        # Normalize Feature Set (the input remains original)
        if self.normalize:
            normalizeFeatures = [
                k
                for k in self.colKeep[feature_set]
                if k not in self.dateCols + self.catCols
            ]
            scaler = pickle.load(
                open(
                    self.mainDir
                    + "Features/Scaler_%s_%i.pickle" % (feature_set, self.version),
                    "rb",
                )
            )
            X[normalizeFeatures] = scaler.transform(X[normalizeFeatures])
            if self.mode == "regression":
                oScaler = pickle.load(
                    open(
                        self.mainDir
                        + "Features/OScaler_%s_%i.pickle" % (feature_set, self.version),
                        "rb",
                    )
                )
                Y = oScaler.transform(Y)

        # Run for regression (different Cross-Validation & worst case (MAE vs ACC))
        if self.mode == "regression":
            gridSearch = GridSearch(
                model, params, cv=KFold(n_splits=self.cvSplits), scoring=self.objective
            )
            results = gridSearch.fit(X, Y)
            results["worst_case"] = results["mean_objective"] + results["std_objective"]
            results = results.sort_values("worst_case")

        # run for classification
        elif self.mode == "classification":
            gridSearch = GridSearch(
                model,
                params,
                cv=StratifiedKFold(n_splits=self.cvSplits),
                scoring=self.objective,
            )
            results = gridSearch.fit(X, Y)
            results["worst_case"] = results["mean_objective"] - results["std_objective"]
            results = results.sort_values("worst_case", ascending=False)

        return results

    def _gridSearchIterationHalvingCV(self, model, params, feature_set):
        """
        Grid search for defined model, parameter set and feature set.
        Contrary to normal function, uses Scikit-Learns HalvingRandomSearchCV. Special halvign algo
        that uses subsets of data for early elimination. Speeds up significantly :)
        """
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingRandomSearchCV

        print(
            "\n[AutoML] Starting Hyperparameter Optimization (halving) for %s on %s features (%i samples, %i features)"
            % (
                type(model).__name__,
                feature_set,
                len(self.X),
                len(self.colKeep[feature_set]),
            )
        )

        # Select data
        X, Y = self.X[self.colKeep[feature_set]], self.Y.values.ravel()

        # Normalize Feature Set (the input remains original)
        if self.normalize:
            normalizeFeatures = [
                k
                for k in self.colKeep[feature_set]
                if k not in self.dateCols + self.catCols
            ]
            scaler = pickle.load(
                open(
                    self.mainDir
                    + "Features/Scaler_%s_%i.pickle" % (feature_set, self.version),
                    "rb",
                )
            )
            X[normalizeFeatures] = scaler.transform(X[normalizeFeatures])
            if self.mode == "regression":
                oScaler = pickle.load(
                    open(
                        self.mainDir
                        + "Features/OScaler_%s_%i.pickle" % (feature_set, self.version),
                        "rb",
                    )
                )
                Y = oScaler.transform(Y)

        # Specify Halving Resource
        resource = "n_samples"
        max_resources = "auto"
        min_resources = int(0.2 * len(X)) if len(X) > 5000 else len(X)
        if model.__module__ == "catboost.core":
            resource = "n_estimators"
            max_resources = 3000
            min_resources = 250
        if (
            model.__module__ == "sklearn.ensemble._bagging"
            or model.__module__ == "xgboost.sklearn"
            or model.__module__ == "lightgbm.sklearn"
            or model.__module__ == "sklearn.ensemble._forest"
        ):
            resource = "n_estimators"
            max_resources = 1500
            min_resources = 50

        # Optimization
        if self.mode == "regression":
            gridSearch = HalvingRandomSearchCV(
                model,
                params,
                resource=resource,
                max_resources=max_resources,
                min_resources=min_resources,
                n_candidates=250,
                cv=KFold(n_splits=self.cvSplits),
                scoring=self.objective,
                factor=3,
                n_jobs=mp.cpu_count() - 1,
                verbose=self.verbose,
            )
            gridSearch.fit(X, Y)
            scikitResults = pd.DataFrame(gridSearch.cv_results_)
            results = pd.DataFrame()
            results[
                ["params", "mean_objective", "std_objective", "mean_time", "std_time"]
            ] = scikitResults[
                [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "mean_fit_time",
                    "std_fit_time",
                ]
            ]
            results["worst_case"] = (
                -results["mean_objective"] - results["std_objective"]
            )
            results = results.sort_values("worst_case")

        if self.mode == "classification":
            gridSearch = HalvingRandomSearchCV(
                model,
                params,
                resource=resource,
                max_resources=max_resources,
                min_resources=min_resources,
                n_candidates=250,
                cv=StratifiedKFold(n_splits=self.cvSplits),
                scoring=self.objective,
                factor=3,
                n_jobs=mp.cpu_count() - 1,
                verbose=self.verbose,
            )
            gridSearch.fit(X, Y)
            scikitResults = pd.DataFrame(gridSearch.cv_results_)
            results = pd.DataFrame()
            results[
                ["params", "mean_objective", "std_objective", "mean_time", "std_time"]
            ] = scikitResults[
                [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "mean_fit_time",
                    "std_fit_time",
                ]
            ]
            results["worst_case"] = results["mean_objective"] - results["std_objective"]
            results = results.sort_values("worst_case")

        # Update resource in params
        if resource != "n_samples":
            for i in range(len(results)):
                results.loc[results.index[i], "params"][resource] = max_resources

        return results

    def _createStacking(self):
        """
        Based on the best performing models, in addition to cheap models based on very different assumptions,
        A stacking model is optimized to enhance/combine the performance of the models.
        """
        # First, the ideal dataset has to be chosen, we're restricted to a single one...
        results = self._sortResults(
            self.results[
                np.logical_and(
                    self.results["type"] == "Hyperparameter Opt",
                    self.results["data_version"] == self.version,
                )
            ]
        )
        feature_set = results["dataset"].value_counts()[0]

    def validate(self, model, feature_set, params=None):
        """
        Just a wrapper for the outside.
        Parameters:
        Model: The model to optimize, either string or class
        Feature Set: String
        (optional) params: Model parameters for which to validate
        """
        assert feature_set in self.colKeep.keys(), "Feature Set not available."

        # Get model
        if isinstance(model, str):
            models = self.Modelling.return_models()
            model = models[
                [i for i in range(len(models)) if type(models[i]).__name__ == model][0]
            ]

        # Get params
        if params is not None:
            params = self._getBestParams(model, feature_set)

        # Run validation
        self._validateResult(model, params, feature_set)

    def _validateResult(self, master_model, params, feature_set):
        print(
            "[AutoML] Validating results for %s (%i %s features) (%s)"
            % (
                type(master_model).__name__,
                len(self.colKeep[feature_set]),
                feature_set,
                params,
            )
        )
        if not os.path.exists(self.mainDir + "Validation/"):
            os.mkdir(self.mainDir + "Validation/")

        # Select data
        X, Y = self.X[self.colKeep[feature_set]], self.Y

        # Normalize Feature Set (the X remains original)
        if self.normalize:
            normalizeFeatures = [
                k
                for k in self.colKeep[feature_set]
                if k not in self.dateCols + self.catCols
            ]
            scaler = pickle.load(
                open(
                    self.mainDir
                    + "Features/Scaler_%s_%i.pickle" % (feature_set, self.version),
                    "rb",
                )
            )
            X[normalizeFeatures] = scaler.transform(X[normalizeFeatures])
            if self.mode == "regression":
                oScaler = pickle.load(
                    open(
                        self.mainDir
                        + "Features/OScaler_%s_%i.pickle" % (feature_set, self.version),
                        "rb",
                    )
                )
                Y[Y.keys()] = oScaler.transform(Y)
                print("(%.1f, %.1f)" % (np.mean(Y), np.std(Y)))
        X, Y = X.to_numpy(), Y.to_numpy().reshape((-1, 1))

        # For Regression
        if self.mode == "regression":

            # Cross-Validation Plots
            fig, ax = plt.subplots(
                math.ceil(self.cvSplits / 2), 2, sharex=True, sharey=True
            )
            fig.suptitle(
                "%i-Fold Cross Validated Predictions - %s"
                % (self.cvSplits, type(master_model).__name__)
            )

            # Initialize iterables
            score = []
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            # Cross Validate
            for i, (t, v) in enumerate(cv.split(X, Y)):
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t].reshape((-1)), Y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(Xt, Yt)

                # Metrics
                score.append(self.scorer(model, Xv, Yv))

                # Plot
                ax[i // 2][i % 2].set_title("Fold-%i" % i)
                ax[i // 2][i % 2].plot(Yv, color="#2369ec")
                ax[i // 2][i % 2].plot(model.predict(Xv), color="#ffa62b", alpha=0.4)

            # Print & Finish plot
            print(
                "[AutoML] %s:        %.2f \u00B1 %.2f"
                % (self.scorer._score_func.__name__, np.mean(score), np.std(score))
            )
            ax[i // 2][i % 2].legend(["Output", "Prediction"])
            plt.show()

        # For BINARY classification
        elif self.mode == "classification" and self.Y[self.target].nunique() == 2:
            # Initiating
            fig, ax = plt.subplots(
                math.ceil(self.cvSplits / 2), 2, sharex=True, sharey=True
            )
            fig.suptitle(
                "%i-Fold Cross Validated Predictions - %s (%s)"
                % (self.cvSplits, type(master_model).__name__, feature_set)
            )
            acc = []
            prec = []
            rec = []
            spec = []
            f1 = []
            aucs = []
            tprs = []
            cm = np.zeros((2, 2))
            mean_fpr = np.linspace(0, 1, 100)

            # Modelling
            cv = StratifiedKFold(n_splits=self.cvSplits)
            for i, (t, v) in enumerate(cv.split(X, Y)):
                n = len(v)
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t].reshape((-1)), Y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(Xt, Yt)
                predictions = model.predict(Xv).reshape((-1))

                # Metrics
                tp = np.logical_and(np.sign(predictions) == 1, Yv == 1).sum()
                tn = np.logical_and(np.sign(predictions) == 0, Yv == 0).sum()
                fp = np.logical_and(np.sign(predictions) == 1, Yv == 0).sum()
                fn = np.logical_and(np.sign(predictions) == 0, Yv == 1).sum()
                acc.append((tp + tn) / n * 100)
                if tp + fp > 0:
                    prec.append(tp / (tp + fp) * 100)
                if tp + fn > 0:
                    rec.append(tp / (tp + fn) * 100)
                if tn + fp > 0:
                    spec.append(tn / (tn + fp) * 100)
                if tp + fp > 0 and tp + fn > 0:
                    f1.append(
                        2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1])
                        if prec[-1] + rec[-1] > 0
                        else 0
                    )
                cm += np.array([[tp, fp], [fn, tn]]) / self.cvSplits

                # Plot
                ax[i // 2][i % 2].plot(Yv, c="#2369ec", alpha=0.6)
                ax[i // 2][i % 2].plot(predictions, c="#ffa62b")
                ax[i // 2][i % 2].set_title("Fold-%i" % i)

            # Results
            print(
                "[AutoML] Accuracy:        %.2f \u00B1 %.2f %%"
                % (np.mean(acc), np.std(acc))
            )
            print(
                "[AutoML] Precision:       %.2f \u00B1 %.2f %%"
                % (np.mean(prec), np.std(prec))
            )
            print(
                "[AutoML] Recall:          %.2f \u00B1 %.2f %%"
                % (np.mean(rec), np.std(rec))
            )
            print(
                "[AutoML] Specificity:     %.2f \u00B1 %.2f %%"
                % (np.mean(spec), np.std(spec))
            )
            print(
                "[AutoML] F1-score:        %.2f \u00B1 %.2f %%"
                % (np.mean(f1), np.std(f1))
            )
            print("[AutoML] Confusion Matrix:")
            print("[AutoML] Pred \ true |  Faulty   |   Healthy      ")
            print(
                "[AutoML]  Faulty     |  %s|  %.1f"
                % (("%.1f" % cm[0, 0]).ljust(9), cm[0, 1])
            )
            print(
                "[AutoML]  Healthy    |  %s|  %.1f"
                % (("%.1f" % cm[1, 0]).ljust(9), cm[1, 1])
            )

            # Check whether plot is possible
            if isinstance(model, sklearn.linear_model.Lasso) or isinstance(
                model, sklearn.linear_model.Ridge
            ):
                return

            # Plot ROC
            fig, ax = plt.subplots(figsize=[12, 8])
            viz = plot_roc_curve(
                model, Xv, Yv, name="ROC fold {}".format(i + 1), alpha=0.3, ax=ax
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            # Adjust plots
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                lw=2,
                color="#ffa62b",
                label="Chance",
                alpha=0.8,
            )
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="#2369ec",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="#729ce9",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )
            ax.set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                title="ROC Curve - %s" % type(master_model).__name__,
            )
            ax.legend(loc="lower right")
            fig.savefig(
                self.mainDir + "Validation/ROC_%s.png" % type(model).__name__,
                format="png",
                dpi=200,
            )

        # For MULTICLASS classification
        elif self.mode == "classification":
            # Initiating
            fig, ax = plt.subplots(
                math.ceil(self.cvSplits / 2), 2, sharex=True, sharey=True
            )
            fig.suptitle(
                "%i-Fold Cross Validated Predictions - %s (%s)"
                % (self.cvSplits, type(master_model).__name__, feature_set)
            )
            n_classes = self.Y[self.target].nunique()
            f1Score = np.zeros((self.cvSplits, n_classes))
            logLoss = np.zeros(self.cvSplits)
            avgAcc = np.zeros(self.cvSplits)

            # Modelling
            cv = StratifiedKFold(n_splits=self.cvSplits)
            for i, (t, v) in enumerate(cv.split(X, Y)):
                n = len(v)
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t].reshape((-1)), Y[v].reshape((-1))
                model = copy.copy(master_model)
                model.set_params(**params)
                model.fit(Xt, Yt)
                predictions = model.predict(Xv).reshape((-1))

                # Metrics
                f1Score[i] = metrics.f1_score(Yv, predictions, average=None)
                avgAcc[i] = metrics.accuracy_score(Yv, predictions)
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(Xv)
                    logLoss[i] = metrics.log_loss(Yv, probabilities)

                # Plot
                ax[i // 2][i % 2].plot(Yv, c="#2369ec", alpha=0.6)
                ax[i // 2][i % 2].plot(predictions, c="#ffa62b")
                ax[i // 2][i % 2].set_title("Fold-%i" % i)

            # Results
            print("F1 scores:")
            print("".join([" Class %i |" % i for i in range(n_classes)]))
            print(
                "".join(
                    [" %.2f ".ljust(9) % f1 + "|" for f1 in np.mean(f1Score, axis=0)]
                )
            )
            print(
                "Average Accuracy: %.2f \u00B1 %.2f" % (np.mean(avgAcc), np.std(avgAcc))
            )
            if hasattr(model, "predict_proba"):
                print(
                    "Log Loss:         %.2f \u00B1 %.2f"
                    % (np.mean(logLoss), np.std(logLoss))
                )

    def _prepareProductionFiles(self, model=None, feature_set=None, params=None):
        if not os.path.exists(self.mainDir + "Production/v%i/" % self.version):
            os.mkdir(self.mainDir + "Production/v%i/" % self.version)
        # Get sorted results for this data version
        results = self._sortResults(
            self.results[self.results["data_version"] == self.version]
        )

        # In the case args are provided
        if model is not None and feature_set is not None:
            # Take name if model instance is given
            if not isinstance(model, str):
                model = type(model).__name__
            if params is None:
                results = self._sortResults(
                    results[
                        np.logical_and(
                            results["model"] == model, results["dataset"] == feature_set
                        )
                    ]
                )
                params = Utils.parseJson(results.iloc[0]["params"])

        # Otherwise find best
        else:
            model = results.iloc[0]["model"]
            feature_set = results.iloc[0]["dataset"]
            params = results.iloc[0]["params"]
            if isinstance(params, str):
                params = Utils.parseJson(params)

        # Notify of results
        print(
            "[AutoML] Preparing Production Env Files for %s, feature set %s"
            % (model, feature_set)
        )
        print("[AutoML] ", params)
        print(
            "[AutoML] %s: %.2f \u00B1 %.2f"
            % (
                self.scorer._score_func.__name__,
                results.iloc[0]["mean_objective"],
                results.iloc[0]["std_objective"],
            )
        )

        # Save Features
        self.bestFeatures = self.colKeep[feature_set]
        json.dump(
            self.bestFeatures,
            open(self.mainDir + "Production/v%i/Features.json" % self.version, "w"),
        )

        # Copy data
        X, Y = self.X[self.bestFeatures], self.Y

        # Save Scalers & Normalize
        if self.normalize:
            # Save
            shutil.copy(
                self.mainDir
                + "Features/Scaler_%s_%i.pickle" % (feature_set, self.version),
                self.mainDir + "Production/v%i/Scaler.pickle" % self.version,
            )
            if self.mode == "regression":
                shutil.copy(
                    self.mainDir
                    + "Features/OScaler_%s_%i.pickle" % (feature_set, self.version),
                    self.mainDir + "Production/v%i/OScaler.pickle" % self.version,
                )

            # Normalize
            normalizeFeatures = [
                k for k in self.bestFeatures if k not in self.dateCols + self.catCols
            ]
            self.bestScaler = pickle.load(
                open(
                    self.mainDir
                    + "Features/Scaler_%s_%i.pickle" % (feature_set, self.version),
                    "rb",
                )
            )
            X = self.bestScaler.transform(X[normalizeFeatures])
            if self.mode == "regression":
                self.bestOScaler = pickle.load(
                    open(
                        self.mainDir
                        + "Features/OScaler_%s_%i.pickle" % (feature_set, self.version),
                        "rb",
                    )
                )
                Y = self.bestOScaler.transform(Y)
            else:
                Y = self.Y[self.target].to_numpy()

        # Cluster Features require additional 'Centers' file
        if any(["dist__" in key for key in self.bestFeatures]):
            shutil.copy(
                self.mainDir + "Features/KMeans_v%i.csv" % self.version,
                self.mainDir + "Production/v%i/KMeans.csv" % self.version,
            )

        # Model
        self.bestModel = [
            mod for mod in self.Modelling.return_models() if type(mod).__name__ == model
        ][0]
        self.bestModel.set_params(**params)
        self.bestModel.fit(X, Y.values.ravel())
        joblib.dump(
            self.bestModel, self.mainDir + "Production/v%i/Model.joblib" % self.version
        )

        # Predict function
        predictCode = self.createPredictFunction(self.customCode)
        with open(self.mainDir + "Production/v%i/Predict.py" % self.version, "w") as f:
            f.write(predictCode)
        with open(self.mainDir + "Production/v%i/__init__.py" % self.version, "w") as f:
            f.write("")

        # Pipeline
        pickle.dump(
            self,
            open(self.mainDir + "Production/v%i/Pipeline.pickle" % self.version, "wb"),
        )
        return

    def testProductionFiles(self, data):
        """
        Test the production files. Not equivalent with predict function as features are stored in class memory.
        """
        print("[AutoML] Testing Production files.")
        # Data Conversion
        X, Y = self._convertData(data)

        # Prediction
        folder = "Production/v%i/" % self.version
        if self.mode == "regression":
            oScaler = pickle.load(open(self.mainDir + folder + "OScaler.pickle", "rb"))
        model = joblib.load(self.mainDir + folder + "Model.joblib")

        # Prepare inputs & Predict
        normalizedPrediction = model.predict(X)

        # Output for Regression
        if self.mode == "regression":
            # Scale
            prediction = oScaler.inverse_transform(normalizedPrediction)
            normalizedOutput = oScaler.transform(Y)

            # Print
            print("Input  ~ (%.1f, %.1f)" % (np.mean(X), np.std(X)))
            print("Output ~ (%.1f, %.1f)" % (np.mean(Y), np.std(Y)))
            print(
                "MAE (normalized):      %.3f"
                % mean_absolute_error(normalizedOutput, normalizedPrediction)
            )
            print("MAE (original):        %.3f" % mean_absolute_error(Y, prediction))
            return prediction

        # Output for Classification
        if self.mode == "classification":
            print("ACC:   %.3f" % average_accuracy(Y, prediction))
            return prediction

    def _errorAnalysis(self):
        # Prepare data
        X, Y = self.bestScaler.transform(self.X[self.bestFeatures]), self.Y
        if self.mode == "regression":
            Y = self.bestOScaler.transform(self.Y)

        # Prediction & error
        prediction = model.predict_proba(X)[:, 1]
        error = Y - prediction

        # Analy
        return ""

    def _convertData(self, data):
        # Load files
        folder = "Production/v%i/" % self.version
        features = json.load(open(self.mainDir + folder + "Features.json", "r"))

        if self.normalize:
            scaler = pickle.load(open(self.mainDir + folder + "Scaler.pickle", "rb"))
            if self.mode == "regression":
                oScaler = pickle.load(
                    open(self.mainDir + folder + "OScaler.pickle", "rb")
                )
        if self.mode == "regression":
            oScaler = pickle.load(open(self.mainDir + folder + "OScaler.pickle", "rb"))

        # Clean data
        data = self.DataProcessing._cleanKeys(data)
        data = self.DataProcessing._convertDataTypes(data)
        data = self.DataProcessing._removeDuplicates(data)
        data = self.DataProcessing._removeOutliers(data)
        data = self.DataProcessing._removeMissingValues(data)
        if (
            data.astype("float32").replace([np.inf, -np.inf], np.nan).isna().sum().sum()
            != 0
        ):
            raise ValueError("Data should not contain NaN or Infs after cleaning!")

        # Save output
        if self.target in data.keys():
            Y = data[self.target].to_numpy().reshape((-1, 1))
        else:
            Y = None

        # Convert Features
        if "KMeans.csv" in os.listdir(self.mainDir + folder):
            k_means = pd.read_csv(self.mainDir + folder + "KMeans.csv")
            X = self.FeatureProcessing.transform(
                data=data, features=features, k_means=k_means
            )
        else:
            X = self.FeatureProcessing.transform(data, features)

        if (
            X.astype("float32").replace([np.inf, -np.inf], np.nan).isna().sum().sum()
            != 0
        ):
            raise ValueError(
                "Data should not contain NaN or Infs after adding features!"
            )

        # Normalize
        if self.normalize:
            X[X.keys()] = scaler.transform(X)

        # Return
        return X, Y

    def predict(self, data):
        """
        Full script to make predictions. Uses 'Production' folder with defined or latest version.
        """
        # Feature Extraction, Selection and Normalization
        model = joblib.load(self.mainDir + "Production/v%i/Model.joblib" % self.version)
        if self.verbose > 0:
            print(
                "[AutoML] Predicting with %s, v%i"
                % (type(model).__name__, self.version)
            )
        X, Y = self._convertData(data)

        # Predict
        if self.mode == "regression":
            if self.normalize:
                predictions = oScaler.inverse_transform(model.predict(X))
            else:
                predictions = model.predict(X)
        if self.mode == "classification":
            try:
                predictions = model.predict_proba(X)[:, 1]
            except:
                predictions = model.predict(X)

        return predictions

    def createPredictFunction(self, custom_code):
        """
        This function returns a string, which can be used to make predictions.
        This is in a predefined format, a Predict class, with a predict funtion taking the arguments
        model: trained sklearn-like class with the .fit() function
        features: list of strings containing all features fed to the model
        scaler: trained sklearn-like class with .transform function
        data: the data to predict on
        Now this function has the arg decoding, which allows custom code injection
        """
        # Check if predict file exists already to increment version
        if os.path.exists(self.mainDir + "Production/v%i/Predict.py" % self.version):
            with open(
                self.mainDir + "Production/v%i/Predict.py" % self.version, "r"
            ) as f:
                predictFile = f.read()
            ind = predictFile.find("self.version = ") + 16
            oldVersion = predictFile[ind : predictFile.find("'", ind)]
            minorVersion = int(oldVersion[oldVersion.find(".") + 1 :])
            version = oldVersion[: oldVersion.find(".") + 1] + str(minorVersion + 1)
        else:
            version = "v%i.0" % self.version
        print("Creating Prediction %s" % version)
        dataProcess = self.DataProcessing.exportFunction()
        featureProcess = self.FeatureProcessing.exportFunction()
        return (
            """import pandas as pd
import numpy as np
import struct, re, copy, os


class Predict(object):

    def __init__(self):
        self.version = '{}'

    def predict(self, model, features, data, **args):
        ''' 
        Prediction function for Amplo's AutoML. 
        This is in a predefined format: 
        - a 'Predict' class, with a 'predict' funtion taking the arguments:        
            model: trained sklearn-like class with the .fit() function
            features: list of strings containing all features fed to the model
            data: the data to predict on
        Note: May depend on additional named arguments within args. 
        '''
        ###############
        # Custom Code #
        ###############""".format(
                version
            )
            + textwrap.indent(custom_code, "    ")
            + dataProcess
            + featureProcess
            + """
        ###########
        # Predict #
        ###########
        mode, normalize = '{}', {}
        
        # Normalize
        if normalize:
            assert 'scaler' in args.keys(), 'When Normalizing=True, scaler needs to be provided in args'
            X = args['scaler'].transform(X)
        
        # Predict
        if mode == 'regression':
            if normalize:
                assert 'o_scaler' in args.keys(), 'When Normalizing=True, o_scaler needs to be provided in args'
                predictions = args['oScaler'].inverse_transform(model.predict(X))
            else:
                predictions = model.predict(X)
        if mode == 'classification':
            try:
                predictions = model.predict_proba(X)[:, 1]
            except:
                predictions = model.predict(X)
        
        return predictions
""".format(
                self.mode, self.normalize
            )
        )


class DataProcessing(object):
    def __init__(
        self,
        target=None,
        num_cols=None,
        date_cols=None,
        cat_cols=None,
        missing_values="interpolate",
        outlier_removal="clip",
        z_score_threshold=4,
        folder="",
        version=1,
        mode="regression",
    ):
        """
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
        """
        # Parameters
        self.folder = folder if len(folder) == 0 or folder[-1] == "/" else folder + "/"
        self.version = version
        self.target = re.sub("[^a-z0-9]", "_", target.lower())
        self.mode = mode
        self.numCols = (
            [re.sub("[^a-z0-9]", "_", nc.lower()) for nc in num_cols]
            if num_cols is not None
            else []
        )
        self.catCols = (
            [re.sub("[^a-z0-9]", "_", cc.lower()) for cc in cat_cols]
            if cat_cols is not None
            else []
        )
        self.dateCols = (
            [re.sub("[^a-z0-9]", "_", dc.lower()) for dc in date_cols]
            if date_cols is not None
            else []
        )
        if self.target in self.numCols:
            self.numCols.remove(self.target)

        # Variables
        self.scaler = None
        self.oScaler = None

        ### Algorithms
        missingValuesImplemented = [
            "remove_rows",
            "remove_cols",
            "interpolate",
            "mean",
            "zero",
        ]
        outlierRemovalImplemented = ["boxplot", "zscore", "clip", "none"]
        if outlier_removal not in outlierRemovalImplemented:
            raise ValueError(
                "Outlier Removal Algorithm not implemented. Should be in "
                + str(outlierRemovalImplemented)
            )
        if missing_values not in missingValuesImplemented:
            raise ValueError(
                "Missing Values Algorithm not implemented. Should be in "
                + str(missingValuesImplemented)
            )
        self.missingValues = missing_values
        self.outlierRemoval = outlier_removal
        self.zScoreThreshold = z_score_threshold

    def clean(self, data):
        print(
            "[Data] Data Cleaning Started, (%i x %i) samples"
            % (len(data), len(data.keys()))
        )
        if len(data[self.target].unique()) == 2:
            self.mode = "classification"

        # Clean
        data = self._cleanKeys(data)
        data = self._removeDuplicates(data)
        data = self._convertDataTypes(data)
        data = self._removeOutliers(data)
        data = self._removeMissingValues(data)
        data = self._removeConstants(data)
        # data = self._normalize(data) --> [Deprecated] Moved right before modelling

        # Finish
        self._store(data)
        print(
            "[Data] Processing completed, (%i x %i) samples returned"
            % (len(data), len(data.keys()))
        )
        return data

    def _cleanKeys(self, data):
        # Clean Keys
        newKeys = {}
        for key in data.keys():
            newKeys[key] = re.sub("[^a-zA-Z0-9 \n\.]", "_", key.lower()).replace(
                "__", "_"
            )
        data = data.rename(columns=newKeys)
        return data

    def _convertDataTypes(self, data):
        # Convert Data Types
        for key in self.dateCols:
            data.loc[:, key] = pd.to_datetime(
                data[key], errors="coerce", infer_datetime_format=True, utc=True
            )
        for key in [
            key
            for key in data.keys()
            if key not in self.dateCols and key not in self.catCols
        ]:
            data.loc[:, key] = pd.to_numeric(
                data[key], errors="coerce", downcast="float"
            )
        for key in self.catCols:
            if key in data.keys():
                dummies = pd.get_dummies(data[key])
                for dummy_key in dummies.keys():
                    dummies = dummies.rename(
                        columns={
                            dummy_key: key
                            + "_"
                            + re.sub("[^a-z0-9]", "_", str(dummy_key).lower())
                        }
                    )
                data = data.drop(key, axis=1).join(dummies)
        if self.target in data.keys():
            data.loc[:, self.target] = pd.to_numeric(data[self.target], errors="coerce")
        return data

    def _removeDuplicates(self, data):
        # Remove Duplicates
        data = data.drop_duplicates()
        data = data.loc[:, ~data.columns.duplicated()]
        return data

    def _removeConstants(self, data):
        # Remove Constants
        data = data.drop(columns=data.columns[data.nunique() == 1])
        return data

    def _removeOutliers(self, data):
        # Remove Outliers
        if self.outlierRemoval == "boxplot":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            for key in Q1.keys():
                data.loc[data[key] < Q1[key] - 1.5 * (Q3[key] - Q1[key]), key] = np.nan
                data.loc[data[key] > Q3[key] + 1.5 * (Q3[key] - Q1[key]), key] = np.nan
        elif self.outlierRemoval == "zscore":
            zScore = (data - data.mean(skipna=True, numeric_only=True)) / data.std(
                skipna=True, numeric_only=True
            )
            data[zScore > self.zScoreThreshold] = np.nan
        elif self.outlierRemoval == "clip":
            data = data.clip(lower=-1e12, upper=1e12)
        return data

    def _removeMissingValues(self, data):
        # Remove Missing Values
        data = data.replace([np.inf, -np.inf], np.nan)
        if self.missingValues == "remove_rows":
            data = data[data.isna().sum(axis=1) == 0]
        elif self.missingValues == "remove_cols":
            data = data.loc[:, data.isna().sum(axis=0) == 0]
        elif self.missingValues == "zero":
            data = data.fillna(0)
        elif self.missingValues == "interpolate":
            ik = np.setdiff1d(data.keys().to_list(), self.dateCols)
            data[ik] = data[ik].interpolate(limit_direction="both")
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif self.missingValues == "mean":
            data = data.fillna(data.mean())
        return data

    def __normalize(self, data):
        """DEPRECATED -- NOT USED"""
        # Organise features that need normalizing, some where deleted
        features = [
            key
            for key in data.keys()
            if key not in self.catCols + self.dateCols + [self.target]
        ]

        # Normalize
        self.scaler = StandardScaler()
        data[features] = self.scaler.fit_transform(data[features])

        # Normalize output for regression
        if self.mode == "regression":
            self.oScaler = StandardScaler()
            data[self.target] = self.oScaler.fit_transform(data[self.target])
        return data

    def _store(self, data):
        # Store cleaned data
        data.to_csv(self.folder + "Cleaned_v%i.csv" % self.version, index_label="index")

    def exportFunction(self):
        functionStrings = [
            inspect.getsource(self._cleanKeys),
            inspect.getsource(self._removeDuplicates).replace("self.", ""),
            inspect.getsource(self._convertDataTypes).replace("self.", ""),
            inspect.getsource(self._removeOutliers).replace("self.", ""),
            inspect.getsource(self._removeMissingValues).replace("self.", ""),
            # inspect.getsource(self._removeConstants),
        ]
        functionStrings = "\n".join(
            [k[k.find("\n") : k.rfind("\n", 0, k.rfind("\n"))] for k in functionStrings]
        )

        return (
            """
        #################
        # Data Cleaning #
        #################
        # Copy vars
        catCols, dateCols, target = {}, {}, '{}'
        outlierRemoval, missingValues, zScoreThreshold = '{}', '{}', '{}'
""".format(
                self.catCols,
                self.dateCols,
                self.target,
                self.outlierRemoval,
                self.missingValues,
                self.zScoreThreshold,
            )
            + functionStrings
        )


class FeatureProcessing(object):
    def __init__(
        self,
        max_lags=10,
        max_diff=2,
        information_threshold=0.99,
        extract_features=True,
        folder="",
        mode=None,
        version="",
    ):
        self.X = None
        self.originalInput = None
        self.Y = None
        self.model = None
        self.mode = mode
        self.threshold = None
        # Register
        self.baseScore = {}
        self.colinearFeatures = None
        self.crossFeatures = None
        self.kMeansFeatures = None
        self.diffFeatures = None
        self.laggedFeatures = None
        # Parameters
        self.maxLags = max_lags
        self.maxDiff = max_diff
        self.informationThreshold = information_threshold
        self.extractFeatures = extract_features
        self.folder = folder if folder == "" or folder[-1] == "/" else folder + "/"
        self.version = version
        # Tests
        assert max_lags >= 0 and max_lags < 50, "Max lags needs to be within [0, 50]"
        assert max_diff >= 0 and max_diff < 3, "Max diff needs to be within [0, 3]"
        assert (
            information_threshold > 0 and information_threshold < 1
        ), "Information threshold needs to be within [0, 1]"
        assert (
            mode is not None
        ), "Mode needs to be specified (regression or classification"

    def extract(self, inputFrame, outputFrame):
        self._cleanAndSet(inputFrame, outputFrame)
        if self.extractFeatures:
            # Manipulate features
            # Order of these doesn't matter, all take originalInput features
            self._removeColinearity()
            self._calcBaseline()
            # self._addMultiFeaturesMP()
            self._addCrossFeatures()
            self._addKMeansFeatures()
            self._addTrigometryFeatures()
            self._addDiffFeatures()
            self._addLaggedFeatures()
        return self.X

    def select(self, inputFrame, outputFrame):
        # Check if not exists
        if os.path.exists(self.folder + "Sets_v%i.json" % self.version):
            return json.load(open(self.folder + "Sets_v%i.json" % self.version, "r"))

        # Execute otherwise
        else:
            # Clean
            self._cleanAndSet(inputFrame, outputFrame)

            # Different Feature Sets
            pps = self._predictivePowerScore()
            rft, rfi = self._randomForestImportance()
            # bp = self._borutaPy()

            # Store & Return
            result = {"PPS": pps, "RFT": rft, "RFI": rfi}  # , 'BP': bp}
            json.dump(result, open(self.folder + "Sets_v%i.json" % self.version, "w"))
            return result

    def transform(self, data, features, **args):
        # Split Features
        crossFeatures = [k for k in features if "__x__" in k or "__d__" in k]
        addFeatures = [k for k in features if "__sub__" in k or "__add__" in k]
        trigoFeatures = [k for k in features if "sin__" in k or "cos__" in k]
        kMeansFeatures = [k for k in features if "dist__" in k]
        diffFeatures = [k for k in features if "__diff__" in k]
        lagFeatures = [k for k in features if "__lag__" in k]
        originalFeatures = [k for k in features if not "__" in k]

        # Make sure centers are provided if kMeansFeatures are nonzero
        if len(kMeansFeatures) != 0:
            if "k_means" not in args:
                raise ValueError(
                    "For K-Means features, the Centers need to be provided."
                )
            k_means = args["k_means"]

        # Fill missing features for normalization
        required = copy.copy(originalFeatures)
        required += [s.split("__")[::2] for s in crossFeatures]
        requried += [s.split("__")[::2] for s in addFeatures]
        required += [k.split("__")[1] for k in trigoFeatures]
        required += [s.split("__diff__")[0] for s in diffFeatures]
        required += [s.split("__lag__")[0] for s in multiFeatures]
        if len(kMeansFeatures) != 0:
            required += [k for k in k_means.keys()]
        for k in [k for k in required if k not in data.keys()]:
            data.loc[:, k] = np.zeros(len(data))

        # Select
        X = data[originalFeatures]

        # Multiplicative features
        for key in crossFeatures:
            if "__x__" in key:
                keyA, keyB = k.split("__x__")
                feature = X[keyA] * X[keyB]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                keyA, keyB = k.split("__d__")
                feature = X[keyA] / X[keyB]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Additive features
        for key in addFeatures:
            if "__sub__" in key:
                keyA, keyB = key.split("__sub__")
                feature = X[keyA] - X[keyB]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                keyA, keyB = key.split("__add__")
                feature = X[keyA] + X[keyB]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Differenced features
        for k in diffFeatures:
            key, diff = k.split("__diff__")
            feature = data[key]
            for i in range(1, diff):
                feature = feature.diff().fillna(0)
            X.loc[:, k] = feature

        # K-Means features
        if len(kMeansFeatures) != 0:
            # Organise data
            temp = copy.copy(data)
            centers = k_means.iloc[:-2]
            means = k_means.iloc[-2]
            stds = k_means.iloc[-1]
            # Normalize
            temp -= means
            temp /= stds
            # Calculate centers
            for key in kMeansFeatures:
                ind = int(key[key.find("dist__") + 6 : key.rfind("_")])
                X.loc[:, key] = np.sqrt(
                    np.square(temp.loc[:, centers.keys()] - centers.iloc[ind]).sum(
                        axis=1
                    )
                )

        # Lagged features
        for k in lagFeatures:
            key, lag = k.split("__lag__")
            X.loc[:, key] = data[key].shift(-int(lag), fill_value=0)

        # Trigonometric features
        for k in trigoFeatures:
            func, key = k.split("__")
            self.X[k] = getattr(np, func)(self.X[key])

        return X

    def exportFunction(self):
        code = inspect.getsource(self.transform)
        return (
            """
        
        ############
        # Features #
        ############"""
            + code[code.find("\n") : code.rfind("\n", 0, code.rfind("\n"))]
        )

    def _cleanAndSet(self, inputFrame, outputFrame):
        assert isinstance(
            inputFrame, pd.DataFrame
        ), "Input supports only Pandas DataFrame"
        assert isinstance(outputFrame, pd.Series), "Output supports only Pandas Series"
        if self.mode == "classification":
            self.model = tree.DecisionTreeClassifier(max_depth=3)
            if outputFrame.nunique() > 2:
                self.mode = "multiclass"
        elif self.mode == "regression":
            self.model = tree.DecisionTreeRegressor(max_depth=3)
        # Bit of necessary data cleaning (shouldn't change anything)
        inputFrame = (
            inputFrame.astype("float32")
            .replace(np.inf, 1e12)
            .replace(-np.inf, -1e12)
            .fillna(0)
            .reset_index(drop=True)
        )
        self.X = copy.copy(inputFrame)
        self.originalInput = copy.copy(inputFrame)
        self.Y = (
            outputFrame.replace([np.inf, -np.inf], 0).fillna(0).reset_index(drop=True)
        )

    def _calcBaseline(self):
        baseline = {}
        for key in self.originalInput.keys():
            baseline[key] = self._analyseFeature(self.X[key])
        self.baseline = pd.DataFrame(baseline).max(axis=1)

    def _analyseFeature(self, feature):
        feature = (
            feature.clip(lower=1e-12, upper=1e12).fillna(0).values.reshape((-1, 1))
        )
        m = copy.copy(self.model)
        m.fit(feature, self.Y)
        if self.mode == "multiclass":
            return [m.score(feature, self.Y, self.Y == i) for i in self.Y.unique()]
        else:
            return [m.score(feature, self.Y)]

    def _acceptFeature(self, candidate):
        if (candidate > self.baseline.values).any():
            return True
        else:
            return False

    def _selectFeatures(self, scores):
        # Convert dict to dataframe
        scores = pd.DataFrame(scores)

        # Select inds
        featuresPerBin = 50
        inds = []
        for score in range(len(scores)):
            inds += [
                i
                for i, k in enumerate(scores.keys())
                if k
                in scores.loc[score]
                .sort_values(ascending=False)
                .keys()[:featuresPerBin]
            ]

        # Return Keys
        return list(scores.keys()[np.unique(inds)])

    def _removeColinearity(self):
        """
        Calculates the Pearson Correlation Coefficient for all input features.
        Those higher than the information threshold are linearly codependent (i.e., describable by y = a x + b)
        These features add little to no information and are therefore removed.
        """
        # Check if not already executed
        if os.path.exists(self.folder + "Colinear_v%i.json" % self.version):
            print("[Features] Loading Colinear features")
            self.colinearFeatures = json.load(
                open(self.folder + "Colinear_v%i.json" % self.version, "r")
            )

        # Else run
        else:
            print("[Features] Analysing colinearity")
            nk = len(self.X.keys())
            norm = (self.X - self.X.mean(skipna=True, numeric_only=True)).to_numpy()
            ss = np.sqrt(np.sum(norm**2, axis=0))
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
            self.colinearFeatures = self.X.keys()[
                np.sum(upper > self.informationThreshold, axis=0) > 0
            ].to_list()
            json.dump(
                self.colinearFeatures,
                open(self.folder + "Colinear_v%i.json" % self.version, "w"),
            )

        self.originalInput = self.originalInput.drop(self.colinearFeatures, axis=1)
        print(
            "[Features] Removed %i Co-Linear features (%.3f %% threshold)"
            % (len(self.colinearFeatures), self.informationThreshold)
        )

    @staticmethod
    def _staticMultiply(model, data, labels, features):
        feature = data[features[0]] * data[features[1]]
        feature = (
            feature.clip(lower=1e-12, upper=1e12).fillna(0).values.reshape((-1, 1))
        )
        model.fit(feature, labels)
        return (
            features[0] + "__x__" + features[1],
            max([model.score(feature, labels, labels == i) for i in labels.unique()]),
        )

    def _addMultiFeaturesMP(self):
        print("[Features] Listing Multiplication Features")
        features = []
        for i, keyA in enumerate(self.originalInput.keys()):
            for j, keyB in enumerate(self.originalInput.keys()):
                if i >= j:
                    continue
                features.append((keyA, keyB))
        print("[Features] Analysing {} Multiplication Features".format(len(features)))
        scores = dict(
            process_map(
                functools.partial(self._staticMultiply, self.model, self.X, self.Y),
                features,
                max_workers=8,
                chunksize=min(100, int(len(features) / 8 / 8)),
            )
        )
        self.multiFeatures = self._selectFeatures(scores)

    def _addCrossFeatures(self):
        """
        Calculates cross-feature features with m and multiplication.
        Should be limited to say ~500.000 features (runs about 100-150 features / second)
        """
        # Check if not already executed
        if (
            os.path.exists(self.folder + "crossFeatures_v%i.json" % self.version)
            and False
        ):
            self.crossFeatures = json.load(
                open(self.folder + "crossFeatures_v%i.json" % self.version, "r")
            )
            print("[Features] Loaded %i cross features" % len(self.crossFeatures))

        # Else, execute
        else:
            print("[Features] Analysing cross features")
            scores = {}
            nKeys = len(self.originalInput.keys())

            # Analyse Cross Features
            for i, keyA in enumerate(tqdm(self.originalInput.keys())):
                acceptedForKeyA = 0
                for j, keyB in enumerate(
                    self.originalInput.keys()[np.random.permutation(nKeys)]
                ):
                    # Skip if they're the same
                    if keyA == keyB:
                        continue
                    # Skip rest if keyA is not useful in first min(50, 30%) (uniform)
                    if acceptedForKeyA == 0 and j > min(50, int(nKeys / 3)):
                        continue

                    # Analyse Division
                    feature = self.X[keyA] / self.X[keyB]
                    score = self._analyseFeature(feature)
                    # Accept or not
                    if self._acceptFeature(score):
                        scores[keyA + "__d__" + keyB] = score
                        acceptedForKeyA += 1

                    # Multiplication i * j == j * i, so skip if j >= i
                    if j >= i:
                        continue

                    # Analyse Multiplication
                    feature = self.X[keyA] * self.X[keyB]
                    score = self._analyseFeature(feature)
                    # Accept or not
                    if self._acceptFeature(score):
                        scores[keyA + "__x__" + keyB] = score
                        acceptedForKeyA += 1

            # Select valuable features
            self.crossFeatures = self._selectFeatures(scores)

        # Add features
        for k in self.crossFeatures:
            if "__x__" in k:
                keyA, keyB = k.split("__x__")
                feature = self.X[keyA] * self.X[keyB]
                self.X[k] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                keyA, keyB = k.split("__d__")
                feature = self.X[keyA] / self.X[keyB]
                self.X[k] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Store
        json.dump(
            self.crossFeatures,
            open(self.folder + "crossFeatures_v%i.json" % self.version, "w"),
        )
        print("[Features] Added %i cross features" % len(self.crossFeatures))

    def _addTrigometryFeatures(self):
        """
        Calculates trigonometry features with sinus, cosines
        """
        # Check if not already executed
        if os.path.exists(self.folder + "trigoFeatures_v%i.json" % self.version):
            self.trigoFeatures = json.load(
                open(self.folder + "trigoFeatures_v%i.json" % self.version, "r")
            )
            print(
                "[Features] Loaded %i trigonometric features" % len(self.trigoFeatures)
            )

        # Else, execute
        else:
            print("[Features] Analysing Trigonometric features")

            scores = {}
            for key in tqdm(self.originalInput.keys()):
                # Sinus feature
                sinFeature = np.sin(self.X[key])
                score = self._analyseFeature(sinFeature)
                if self._acceptFeature(score):
                    score["sin__" + key] = score

                # Cosinus feature
                cosFeature = np.cos(self.X[key])
                score = self._analyseFeature(cosFeature)
                if self._acceptFeature(score):
                    scores["cos__" + key]

        # Select valuable features
        self.trigoFeatures = self._selectFeatures(scores)

        # Add features
        for k in self.trigoFeatures:
            func, key = k.split("__")
            self.X[k] = getattr(np, func)(self.X[key])

        # Store
        json.dump(
            self.trigoFeatures,
            open(self.folder + "trigoFeatures_v%i.json" % self.version, "w"),
        )
        print("[Features] Added %i trigonometric features" % len(self.trigoFeatures))

    def _addAdditiveFeatures(self):
        """
        Calculates simple additive and subtractive features
        """
        # Load if available
        if os.path.exists(self.folder + "addFeatures_v%i.json" % self.version):
            self.addFeatures = json.load(
                open(self.folder + "addFeatures_v%i.json" % self.version, "r")
            )
            print("[Features] Loaded %i additive features" % len(self.addFeatures))

        # Else, execute
        else:
            scores = {}
            nKeys = len(self.originalInput.keys())
            for i, keyA in enumerate(self.originalInput.keys()):
                acceptedForKeyA = 0
                for j, keyB in enumerate(
                    self.originalInput.keys()[np.random.permutation(nKeys)]
                ):
                    # Skip if they're the same
                    if keyA == keyB:
                        continue
                    # Skip rest if keyA is not useful in first min(50, 30%) (uniform)
                    if acceptedForKeyA == 0 and j > min(50, int(nKeys / 3)):
                        continue

                    # Subtracting feature
                    feature = self.input[keyA] - self.input[keyB]
                    score = self._analyseFeature(feature)
                    if self._acceptFeature(score):
                        scores[keyA + "__sub__" + keyB] = score
                        acceptedForKeyA += 1

                    # A + B == B + A, so skip if i > j
                    if j > i:
                        continue

                    # Additive feature
                    feature = self.input[keyA] + self.input[keyB]
                    score = self._analyseFeature(feature)
                    if self._acceptFeature(score):
                        scores[keyA + "__add__" + keyB] = score
                        acceptedForKeyA += 1

        # Select valuable Features
        self.addFeatures = self._selectFeatures(scores)

        # Add features
        for key in self.addFeatures:
            if "__sub__" in key:
                keyA, keyB = key.split("__sub__")
                feature = self.X[keyA] - self.X[keyB]
                self.X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                keyA, keyB = key.split("__add__")
                feature = self.X[keyA] + self.X[keyB]
                self.X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # store
        json.dump(
            self.addFeatures,
            open(self.folder + "addFeatures_v%i.json" % self.version, "w"),
        )
        print("[Features] Added %i additive features" % len(self.addFeatures))

    def _addKMeansFeatures(self):
        """
        Analyses the correlation of k-means features.
        k-means is a clustering algorithm which clusters the data.
        The distance to each cluster is then analysed.
        """
        # Check if not exist
        if os.path.exists(self.folder + "K-MeansFeatures_v%i.json" % self.version):
            # Load features and cluster size
            self.kMeansFeatures = json.load(
                open(self.folder + "K-MeansFeatures_v%i.json" % self.version, "r")
            )
            kMeansData = pd.read_csv(self.folder + "KMeans_v%i.csv" % self.version)
            print("[Features] Loaded %i K-Means features" % len(self.kMeansFeatures))

            # Prepare data
            data = copy.copy(self.originalInput)
            centers = kMeansData.iloc[:-2]
            means = kMeansData.iloc[-2]
            stds = kMeansData.iloc[-1]
            data -= means
            data /= stds

            # Add them
            for key in self.kMeansFeatures:
                ind = int(key[key.find("dist__") + 6 : key.rfind("_")])
                self.X[key] = np.sqrt(np.square(data - centers.iloc[ind]).sum(axis=1))

        # If not executed, analyse all
        else:
            print("[Features] Calculating and Analysing K-Means features")
            # Prepare data
            data = copy.copy(self.originalInput)
            means = data.mean()
            stds = data.std()
            stds[stds == 0] = 1
            data -= means
            data /= stds

            # Determine clusters
            clusters = min(
                max(int(np.log10(len(self.originalInput)) * 8), 8),
                len(self.originalInput.keys()),
            )
            kmeans = cluster.MiniBatchKMeans(n_clusters=clusters)
            columnNames = ["dist__%i_%i" % (i, clusters) for i in range(clusters)]
            distances = pd.DataFrame(
                columns=columnNames, data=kmeans.fit_transform(data)
            )
            distances = distances.clip(lower=1e-12, upper=1e12).fillna(0)

            # Analyse correlation
            scores = {}
            for key in tqdm(distances.keys()):
                score = self._analyseFeature(distances[key])
                if self._acceptFeature(score):
                    scores[key] = score

            # Add the valuable features
            self.kMeansFeatures = self._selectFeatures(scores)
            for k in self.kMeansFeatures:
                self.X[k] = distances[k]

            # Create output
            centers = pd.DataFrame(
                columns=self.originalInput.keys(), data=kmeans.cluster_centers_
            )
            centers = centers.append(means, ignore_index=True)
            centers = centers.append(stds, ignore_index=True)
            centers.to_csv(self.folder + "KMeans_v%i.csv" % self.version, index=False)
            json.dump(
                self.kMeansFeatures,
                open(self.folder + "K-MeansFeatures_v%i.json" % self.version, "w"),
            )
            print(
                "[Features] Added %i K-Means features (%i clusters)"
                % (len(self.kMeansFeatures), clusters)
            )

    def _addDiffFeatures(self):
        """
        Analyses whether the differenced signal of the data should be included.
        """

        # Check if we're allowed
        if self.maxDiff == 0:
            print("[Features] Differenced features skipped, max diff = 0")
            return

        # Check if exist
        if os.path.exists(self.folder + "diffFeatures_v%i.json" % self.version):
            self.diffFeatures = json.load(
                open(self.folder + "diffFeatures_v%i.json" % self.version, "r")
            )
            print("[Features] Loaded %i differenced features" % len(self.diffFeatures))

        # If not exist, execute
        else:
            print("[Features] Analysing differenced features")
            # Copy data so we can diff without altering original data
            keys = self.originalInput.keys()
            diffInput = copy.copy(self.originalInput)

            # Calculate scores
            scores = {}
            for diff in tqdm(range(1, self.maxDiff + 1)):
                diffInput = diffInput.diff().fillna(0)
                for key in keys:
                    score = self._analyseFeature(diffInput[key])
                    if self._acceptFeature(score):
                        scores[key + "__diff__%i" % diff] = score

            # Select the valuable features
            self.diffFeatures = self._selectFeatures(scores)
            print("[Features] Added %i differenced features" % len(self.diffFeatures))

        # Add Differenced Features
        for k in self.diffFeatures:
            key, diff = k.split("__diff__")
            feature = self.X[key]
            for i in range(1, diff):
                feature = feature.diff().clip(lower=1e-12, upper=1e12).fillna(0)
            self.X[k] = feature
        json.dump(
            self.diffFeatures,
            open(self.folder + "diffFeatures_v%i.json" % self.version, "w"),
        )

    def _addLaggedFeatures(self):
        """
        Analyses the correlation of lagged features (value of sensor_x at t-1 to predict target at t)
        """
        # Check if allowed
        if self.maxLags == 0:
            print("[Features] Lagged features skipped, max lags = 0")
            return

        # Check if exists
        if os.path.exists(self.folder + "laggedFeatures_v%i.json" % self.version):
            self.laggedFeatures = json.load(
                open(self.folder + "laggedFeatures_v%i.json" % self.version, "r")
            )
            print("[Features] Loaded %i lagged features" % len(self.laggedFeatures))

        # Else execute
        else:
            print("[Features] Analysing lagged features")
            keys = self.originalInput.keys()
            scores = {}
            for lag in tqdm(range(1, self.maxLags)):
                for key in keys:
                    score = self._analyseFeature(self.X[key].shift(-1))
                    if self._acceptFeature(score):
                        scores[key + "__lag__%i" % lag] = score

            # Select
            self.laggedFeatures = self._selectFeatures(scores)
            print("[Features] Added %i lagged features" % len(self.laggedFeatures))

        # Add selected
        for k in self.laggedFeatures:
            key, lag = k.split("__lag__")
            self.X[k] = self.originalInput[key].shift(-int(lag), fill_value=0)

        # Store
        json.dump(
            self.laggedFeatures,
            open(self.folder + "laggedFeatures_v%i.json" % self.version, "w"),
        )

    def _predictivePowerScore(self):
        """
        Calculates the Predictive Power Score (https://github.com/8080labs/ppscore)
        Assymmetric correlation based on single decision trees trained on 5.000 samples with 4-Fold validation.
        """
        print("[Features] Determining features with PPS")
        data = self.X.copy()
        data["target"] = self.Y.copy()
        pp_score = pps.predictors(data, "target")
        pp_cols = pp_score["x"][pp_score["ppscore"] != 0].to_list()
        print(
            "[Features] Selected %i features with Predictive Power Score" % len(pp_cols)
        )
        return pp_cols

    def _randomForestImportance(self):
        """
        Calculates Feature Importance with Random Forest, aka Mean Decrease in Gini Impurity
        Symmetric correlation based on multiple features and multiple trees ensemble
        """
        print("[Features] Determining features with RF")
        if self.mode == "regression":
            rf = RandomForestRegressor().fit(self.X, self.Y)
        elif self.mode == "classification" or self.mode == "multiclass":
            rf = RandomForestClassifier().fit(self.X, self.Y)
        fi = rf.feature_importances_
        sfi = fi.sum()
        ind = np.flip(np.argsort(fi))
        # Info Threshold
        ind_keep = [
            ind[i]
            for i in range(len(ind))
            if fi[ind[:i]].sum() <= self.informationThreshold * sfi
        ]
        thresholded = self.X.keys()[ind_keep].to_list()
        ind_keep = [ind[i] for i in range(len(ind)) if fi[i] > sfi / 100]
        increment = self.X.keys()[ind_keep].to_list()
        print("[Features] Selected %i features with RF thresholded" % len(thresholded))
        print("[Features] Selected %i features with RF increment" % len(increment))
        return thresholded, increment

    def _borutaPy(self):
        print("[Features] Determining features with Boruta")
        if self.mode == "regression":
            rf = RandomForestRegressor()
        elif self.mode == "classification" or self.mode == "multiclass":
            rf = RandomForestClassifier()
        selector = BorutaPy(rf, n_estimators="auto", verbose=0)
        selector.fit(self.X.to_numpy(), self.Y.to_numpy())
        bp_cols = self.X.keys()[selector.support_].to_list()
        print("[Features] Selected %i features with Boruta" % len(bp_cols))
        return bp_cols


class ExploratoryDataAnalysis(object):
    def __init__(
        self,
        data,
        Y=None,
        plot_timeplots=True,
        plot_boxplots=False,
        plot_missing_values=True,
        plot_seasonality=False,
        plot_colinearity=True,
        plot_differencing=False,
        plot_signal_correlations=False,
        plot_feature_importance=True,
        plot_scatterplots=False,
        differ=0,
        pretag="",
        maxSamples=10000,
        seasonPeriods=[24 * 60, 7 * 24 * 60],
        lags=60,
        skip_completed=True,
        folder="",
        version=None,
    ):
        """
        Doing all the fun EDA in an automized script :)
        :param data: Pandas Dataframe
        :param output: Pandas series of the output
        :param seasonPeriods: List of periods to check for seasonality
        :param lags: Lags for (P)ACF and
        """
        assert isinstance(data, pd.DataFrame)

        # Running booleans
        self.plot_timeplots = plot_timeplots
        self.plot_boxplots = plot_boxplots
        self.plot_missing_values = plot_missing_values
        self.plot_seasonality = plot_seasonality
        self.plot_colinearity = plot_colinearity
        self.plot_differencing = plot_differencing
        self.plot_signal_correlations = plot_signal_correlations
        self.plot_feature_importance = plot_feature_importance
        self.plot_scatterplots = plot_scatterplots

        # Register data
        self.data = data.astype("float32").fillna(0)
        if Y is not None:
            assert isinstance(Y, pd.DataFrame) or isnstance(Y, pd.Series)
            if isinstance(Y, pd.DataFrame):
                Y = Y[Y.keys()[0]]
            self.Y = Y.astype("float32").fillna(0)
            if self.Y.nunique() == 2:
                print("[AutoML] Mode set to classification.")
                self.mode = "classification"
                if set(self.Y.values) != {0, 1}:
                    assert (
                        1 in self.Y.values
                    ), "Ambiguous classes (either {0, 1} or {-1, 1})"
                    self.Y.loc[self.Y.values != 1] = 0
            else:
                print("[AutoML] Mode set to regression.")
                self.mode = "regression"
        else:
            self.mode = None

        # General settings
        self.seasonPeriods = seasonPeriods
        self.maxSamples = maxSamples  # Timeseries
        self.differ = differ  # Correlations
        self.lags = lags  # Correlations

        # Storage settings
        self.tag = pretag
        self.version = version if version is not None else 0
        self.folder = folder if folder == "" or folder[-1] == "/" else folder + "/"
        self.skip = skip_completed

        # Create Base folder
        if not os.path.exists(self.folder + "EDA/"):
            os.mkdir(self.folder + "EDA")
        self.folder += "EDA/"
        self.run()

    def run(self):
        # Run all functions
        if self.mode == "classification":
            self._runClassification()
        else:
            self._runRegression()

    def _runClassification(self):
        print("[EDA] Generating Missing Values Plot")
        self.missingValues()
        print("[EDA] Generating Timeplots")
        self.timeplots()
        print("[EDA] Generating Boxplots")
        self.boxplots()
        if self.Y is not None:
            print("[EDA] Generating SHAP plot")
            self.SHAP()
            print("[EDA] Generating Feature Ranking Plot")
            self.featureRanking()
            print("[EDA] Predictive Power Score Plot")
            self.predictivePowerScore()

    def _runRegression(self):
        print("[EDA] Generating Missing Values Plot")
        self.missingValues()
        print("[EDA] Generating Timeplots")
        self.timeplots()
        print("[EDA] Generating Boxplots")
        self.boxplots()
        self.seasonality()
        print("[EDA] Generating Colinearity Plots")
        self.colinearity()
        print("[EDA] Generating Diff Var Plot")
        self.differencing()
        print("[EDA] Generating ACF Plots")
        self.completeAutoCorr()
        print("[EDA] Generating PACF Plots")
        self.partialAutoCorr()
        if self.Y is not None:
            print("[EDA] Generating CCF Plots")
            self.crossCorr()
            print("[EDA] Generating Scatter plots")
            self.scatters()
            print("[EDA] Generating SHAP plot")
            self.SHAP()
            print("[EDA] Generating Feature Ranking Plot")
            self.featureRanking()
            print("[EDA] Predictive Power Score Plot")
            self.predictivePowerScore()

    def missingValues(self):
        if self.plot_missing_values:
            # Create folder
            if not os.path.exists(self.folder + "MissingValues/"):
                os.mkdir(self.folder + "MissingValues/")

            # Skip if exists
            if self.tag + "v%i.png" % self.version in os.listdir(
                self.folder + "MissingValues/"
            ):
                return

            # Plot
            import missingno

            ax = missingno.matrix(self.data, figsize=[24, 16])
            fig = ax.get_figure()
            fig.savefig(self.folder + "MissingValues/v%i.png" % self.version)

    def boxplots(self):
        if self.plot_boxplots:
            # Create folder
            if not os.path.exists(self.folder + "Boxplots/v%i/" % self.version):
                os.makedirs(self.folder + "Boxplots/v%i/" % self.version)

            # Iterate through vars
            for key in tqdm(self.data.keys()):

                # Skip if existing
                if self.tag + key + ".png" in os.listdir(
                    self.folder + "Boxplots/v%i/" % self.version
                ):
                    continue

                # Figure prep
                fig = plt.figure(figsize=[24, 16])
                plt.title(key)

                # Classification
                if self.mode == "classification":
                    plt.boxplot(
                        [
                            self.data.loc[self.Y == 1, key],
                            self.data.loc[self.Y == -1, key],
                        ],
                        labels=["Faulty", "Healthy"],
                    )
                    plt.legend(["Faulty", "Healthy"])

                # Regression
                if self.mode == "regression":
                    plt.boxplot(self.data[key])

                # Store & Close
                fig.savefig(
                    self.folder
                    + "Boxplots/v%i/" % self.version
                    + self.tag
                    + key
                    + ".png",
                    format="png",
                    dpi=300,
                )
                plt.close()

    def timeplots(self):
        if self.plot_timeplots:
            # Create folder
            if not os.path.exists(self.folder + "Timeplots/v%i/" % self.version):
                os.makedirs(self.folder + "Timeplots/v%i/" % self.version)

            # Set matplot limit
            matplotlib.use("Agg")
            matplotlib.rcParams["agg.path.chunksize"] = 200000

            # Undersample
            ind = np.linspace(0, len(self.data) - 1, self.maxSamples).astype("int")
            data, Y = self.data.iloc[ind], self.Y.iloc[ind]

            # Iterate through features
            for key in tqdm(data.keys()):
                # Skip if existing
                if self.tag + key + ".png" in os.listdir(
                    self.folder + "Timeplots/v%i/" % self.version
                ):
                    continue

                # Figure preparation
                fig = plt.figure(figsize=[24, 16])
                plt.title(key)

                # Plot
                if self.mode == "classification":
                    cm = plt.get_cmap("bwr")
                else:
                    cm = plt.get_cmap("summer")
                nmOutput = (Y - Y.min()) / (Y.max() - Y.min())
                plt.scatter(data.index, data[key], c=cm(nmOutput), alpha=0.3)

                # Store & Close
                fig.savefig(
                    self.folder
                    + "Timeplots/v%i/" % self.version
                    + self.tag
                    + key
                    + ".png",
                    format="png",
                    dpi=100,
                )
                plt.close(fig)

    def seasonality(self):
        if self.plot_seasonality:
            # Create folder
            if not os.path.exists(self.folder + "Seasonality/"):
                os.mkdir(self.folder + "Seasonality/")

            # Iterate through features
            for key in tqdm(self.data.keys()):
                for period in self.seasonPeriods:
                    if self.tag + key + "_v%i.png" % self.version in os.listdir(
                        self.folder + "Seasonality/"
                    ):
                        continue
                    seasonality = STL(self.data[key], period=period).fit()
                    fig = plt.figure(figsize=[24, 16])
                    plt.plot(range(len(self.data)), self.data[key])
                    plt.plot(range(len(self.data)), seasonality)
                    plt.title(key + ", period=" + str(period))
                    fig.savefig(
                        self.folder
                        + "Seasonality/"
                        + self.tag
                        + str(period)
                        + "/"
                        + key
                        + "_v%i.png" % self.version,
                        format="png",
                        dpi=300,
                    )
                    plt.close()

    def colinearity(self):
        if self.plot_colinearity:
            # Create folder
            if not os.path.exists(self.folder + "Colinearity/v%i/"):
                os.makedirs(self.folder + "Colinearity/v%i/")

            # Skip if existing
            if self.tag + "Minimum_Representation.png" in os.listdir(
                self.folder + "Colinearity/v%i/" % self.version
            ):
                return

            # Plot thresholded matrix
            threshold = 0.95
            fig = plt.figure(figsize=[24, 16])
            plt.title("Colinearity matrix, threshold %.2f" % threshold)
            sns.heatmap(abs(self.data.corr()) < threshold, annot=False, cmap="Greys")
            fig.savefig(
                self.folder
                + "Colinearity/v%i/" % self.version
                + self.tag
                + "Matrix.png",
                format="png",
                dpi=300,
            )

            # Minimum representation
            corr_mat = self.data.corr().abs()
            upper = corr_mat.where(
                np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool)
            )
            col_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            minimal_rep = self.data.drop(self.data[col_drop], axis=1)
            fig = plt.figure(figsize=[24, 16])
            sns.heatmap(abs(minimal_rep.corr()) < threshold, annot=False, cmap="Greys")
            fig.savefig(
                self.folder
                + "Colinearity/v%i/" % self.version
                + self.tag
                + "Minimum_Representation.png",
                format="png",
                dpi=300,
            )

    def differencing(self):
        if self.plot_differencing:
            # Create folder
            if not os.path.exists(self.folder + "Lags/"):
                os.mkdir(self.folder + "Lags/")

            # Skip if existing
            if self.tag + "Variance.png" in os.listdir(self.folder + "Lags/"):
                return

            # Setup
            max_lags = 4
            varVec = np.zeros((max_lags, len(self.data.keys())))
            diffData = self.data / np.sqrt(self.data.var())

            # Calculate variance per lag
            for i in range(max_lags):
                varVec[i, :] = diffData.var()
                diffData = diffData.diff(1)[1:]

            # Plot
            fig = plt.figure(figsize=[24, 16])
            plt.title("Variance for different lags")
            plt.plot(varVec)
            plt.xlabel("Lag")
            plt.yscale("log")
            plt.ylabel("Average variance")
            fig.savefig(
                self.folder + "Lags/" + self.tag + "Variance.png", format="png", dpi=300
            )

    def completeAutoCorr(self):
        if self.plot_signal_correlations:
            # Create folder
            if not os.path.exists(self.folder + "Correlation/ACF/"):
                os.makedirs(self.folder + "Correlation/ACF/")

            # Difference data
            diffData = copy.copy(self.data)
            for i in range(self.differ):
                diffData = diffData.diff(1)[1:]

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + "_differ_" + str(
                    self.differ
                ) + "_v%i.png" % self.version in os.listdir(
                    self.folder + "Correlation/ACF/"
                ):
                    continue

                # Plot
                fig = plot_acf(diffData[key], fft=True)
                plt.title(key)
                fig.savefig(
                    self.folder
                    + "Correlation/ACF/"
                    + self.tag
                    + key
                    + "_differ_"
                    + str(self.differ)
                    + "_v%i.png" % self.version,
                    format="png",
                    dpi=300,
                )
                plt.close()

    def partialAutoCorr(self):
        if self.plot_signal_correlations:
            # Create folder
            if not os.path.exists(self.folder + "Correlation/PACF/"):
                os.makedirs(self.folder + "Correlation/PACF/")

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + "_differ_" + str(
                    self.differ
                ) + "_v%i.png" % self.version in os.listdir(
                    self.folder + "EDA/Correlation/PACF/"
                ):
                    continue

                # Plot
                try:
                    fig = plot_pacf(self.data[key])
                    fig.savefig(
                        self.folder
                        + "EDA/Correlation/PACF/"
                        + self.tag
                        + key
                        + "_differ_"
                        + str(self.differ)
                        + "_v%i.png" % self.version,
                        format="png",
                        dpi=300,
                    )
                    plt.title(key)
                    plt.close()
                except:
                    continue

    def crossCorr(self):
        if self.plot_signal_correlations:
            # Create folder
            if not os.path.exists(self.folder + "Correlation/Cross/"):
                os.makedirs(self.folder + "Correlation/Cross/")

            # Prepare
            folder = "Correlation/Cross/"
            Y = self.Y.to_numpy().reshape((-1))

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if self.tag + key + "_differ_" + str(
                    self.differ
                ) + "_v%i.png" % self.version in os.listdir(self.folder + folder):
                    continue

                # Plot
                try:
                    fig = plt.figure(figsize=[24, 16])
                    plt.xcorr(self.data[key], Y, maxlags=self.lags)
                    plt.title(key)
                    fig.savefig(
                        self.folder
                        + folder
                        + self.tag
                        + key
                        + "_differ_"
                        + str(self.differ)
                        + "_v%i.png" % self.version,
                        format="png",
                        dpi=300,
                    )
                    plt.close()
                except:
                    continue

    def scatters(self):
        if self.plot_scatterplots:
            # Create folder
            if not os.path.exists(self.folder + "Scatters/v%i/" % self.version):
                os.makedirs(self.folder + "Scatters/v%i/" % self.version)

            # Iterate through features
            for key in tqdm(self.data.keys()):
                # Skip if existing
                if "{}{}.png".format(self.tag, key) in os.listdir(
                    self.folder + "Scatters/v%i/" % self.version
                ):
                    continue

                # Plot
                fig = plt.figure(figsize=[24, 16])
                plt.scatter(self.Y, self.data[key], alpha=0.2)
                plt.ylabel(key)
                plt.xlabel("Output")
                plt.title("Scatterplot " + key + " - output")
                fig.savefig(
                    self.folder
                    + "Scatters/v%i/" % self.version
                    + self.tag
                    + key
                    + ".png",
                    format="png",
                    dpi=100,
                )
                plt.close(fig)

    def SHAP(self, args={}):
        if self.plot_feature_importance:
            # Create folder
            if not os.path.exists(self.folder + "Features/v%i/" % self.version):
                os.makedirs(self.folder + "Features/v%i/" % self.version)

            # Skip if existing
            if self.tag + "SHAP.png" in os.listdir(
                self.folder + "Features/v%i/" % self.version
            ):
                return

            # Create model
            if self.mode == "classification":
                model = RandomForestClassifier(**args).fit(self.data, self.Y)
            else:
                model = RandomForestRegressor(**args).fit(self.data, self.Y)

            # Calculate SHAP values
            import shap

            shap_values = shap.TreeExplainer(model).shap_values(self.data)

            # Plot
            fig = plt.figure(figsize=[8, 32])
            plt.subplots_adjust(left=0.4)
            shap.summary_plot(shap_values, self.data, plot_type="bar")
            fig.savefig(
                self.folder + "Features/v%i/" % self.version + self.tag + "SHAP.png",
                format="png",
                dpi=300,
            )

    def featureRanking(self, **args):
        if self.plot_feature_importance:
            # Create folder
            if not os.path.exists(self.folder + "Features/v%i/" % self.version):
                os.mkdir(self.folder + "Features/v%i/" % self.version)

            # Skip if existing
            if self.tag + "RF.png" in os.listdir(
                self.folder + "Features/v%i/" % self.version
            ):
                return

            # Create model
            if self.mode == "classification":
                model = RandomForestClassifier(**args).fit(self.data, self.Y)
            else:
                model = RandomForestRegressor(**args).fit(self.data, self.Y)

            # Plot
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ind = np.argsort(model.feature_importances_)
            plt.barh(
                list(self.data.keys()[ind])[-15:],
                width=model.feature_importances_[ind][-15:],
                color="#2369ec",
            )
            fig.savefig(
                self.folder + "Features/v%i/" % self.version + self.tag + "RF.png",
                format="png",
                dpi=300,
            )
            plt.close()

            # Store results
            results = pd.DataFrame(
                {"x": self.data.keys(), "score": model.feature_importances_}
            )
            results.to_csv(
                self.folder + "Features/v%i/" % self.version + self.tag + "RF.csv"
            )

    def predictivePowerScore(self):
        if self.plot_feature_importance:
            # Create folder
            if not os.path.exists(self.folder + "Features/v%i/" % self.version):
                os.mkdir(self.folder + "Features/v%i/" % self.version)

            # Skip if existing
            if self.tag + "Ppscore.png" in os.listdir(
                self.folder + "Features/v%i/" % self.version
            ):
                return

            # Calculate PPS
            data = self.data.copy()
            if isinstance(self.Y, pd.core.series.Series):
                data.loc[:, "target"] = self.Y
            elif isinstance(self.Y, pd.DataFrame):
                data.loc[:, "target"] = self.Y.loc[:, self.Y.keys()[0]]
            pp_score = pps.predictors(data, "target").sort_values("ppscore")

            # Plot
            fig, ax = plt.subplots(figsize=[4, 6], constrained_layout=True)
            plt.subplots_adjust(left=0.5, top=1, bottom=0)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            plt.barh(
                pp_score["x"][-15:], width=pp_score["ppscore"][-15:], color="#2369ec"
            )
            fig.savefig(
                self.folder + "Features/v%i/" % self.version + self.tag + "Ppscore.png",
                format="png",
                dpi=400,
            )
            plt.close()

            # Store results
            pp_score.to_csv(self.folder + "Features/v%i/pp_score.csv" % self.version)


class Modelling(object):
    def __init__(
        self,
        mode="regression",
        shuffle=False,
        plot=False,
        scoring=None,
        folder="models/",
        n_splits=3,
        dataset=0,
        store_models=False,
        store_results=True,
    ):
        self.scoring = scoring
        self.mode = mode
        self.shuffle = shuffle
        self.plot = plot
        self.samples = None
        self.acc = []
        self.std = []
        self.cvSplits = n_splits
        self.dataset = str(dataset)
        self.store_results = store_results
        self.store_models = store_models
        self.folder = folder if folder[-1] == "/" else folder + "/"

    def fit(self, X, Y):
        self.samples = len(Y)
        if self.mode == "regression":
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            return self._fit(X, Y, cv)
        if self.mode == "classification":
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            return self._fit(X, Y, cv)

    def return_models(self):
        from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree
        from sklearn.experimental import enable_hist_gradient_boosting

        models = []

        if self.mode == "classification":
            # self.scorer._factory_args().find('True')

            models.append(linear_model.RidgeClassifier())
            # lasso = linear_model.Lasso()
            # sgd = linear_model.SGDClassifier()
            # knc = neighbors.KNeighborsClassifier()
            # dtc = tree.DecisionTreeClassifier()
            # ada = ensemble.AdaBoostClassifier()
            models.append(
                catboost.CatBoostClassifier(
                    verbose=0, n_estimators=1000, allow_writing_files=False
                )
            )
            if self.samples < 25000:
                models.append(svm.SVC(kernel="rbf"))
                models.append(ensemble.BaggingClassifier())
                models.append(ensemble.GradientBoostingClassifier())
                models.append(
                    xgboost.XGBClassifier(
                        n_estimators=250, verbosity=0, use_label_encoder=False
                    )
                )
            else:
                models.append(ensemble.HistGradientBoostingClassifier())
                models.append(
                    lightgbm.LGBMClassifier(
                        n_estimators=250, verbose=-1, force_row_wise=True
                    )
                )
            models.append(ensemble.RandomForestClassifier())
            # mlp = neural_network.MLPClassifier()

        elif self.mode == "regression":
            models.append(linear_model.Ridge())
            # lasso = linear_model.Lasso()
            # sgd = linear_model.SGDRegressor()
            # knr = neighbors.KNeighborsRegressor()
            # dtr = tree.DecisionTreeRegressor()
            # ada = ensemble.AdaBoostRegressor()
            models.append(
                catboost.CatBoostRegressor(
                    verbose=0, n_estimators=1000, allow_writing_files=False
                )
            )
            models.append(ensemble.BaggingRegressor())
            if self.samples < 25000:
                models.append(svm.SVR(kernel="rbf"))
                models.append(ensemble.GradientBoostingRegressor())
                models.append(xgboost.XGBRegressor(n_estimators=250, verbosity=0))
            else:
                models.append(ensemble.HistGradientBoostingRegressor())
                models.append(
                    lightgbm.LGBMRegressor(
                        n_estimators=250, verbose=-1, force_row_wise=True
                    )
                )
            models.append(ensemble.RandomForestRegressor())
            # mlp = neural_network.MLPRegressor())

        # Filter predict_proba models
        models = [m for m in models if hasattr(m, "predict_proba")]

        return models

    def _fit(self, X, Y, cross_val):
        # Convert to NumPy
        X = np.array(X)
        Y = np.array(Y).ravel()

        # Data
        print(
            "[Modelling] Splitting data (shuffle=%s, splits=%i, features=%i)"
            % (str(self.shuffle), self.cvSplits, len(X[0]))
        )

        if self.store_results and "Initial_Models.csv" in os.listdir(self.folder):
            results = pd.read_csv(self.folder + "Initial_Models.csv")
        else:
            results = pd.DataFrame(
                columns=[
                    "date",
                    "model",
                    "dataset",
                    "params",
                    "mean_objective",
                    "std_objective",
                    "mean_time",
                    "std_time",
                ]
            )

        # Models
        self.models = self.return_models()

        # Loop through models
        for master_model in self.models:
            # Check first if we don't already have the results
            ind = np.where(
                np.logical_and(
                    np.logical_and(
                        results["model"] == type(master_model).__name__,
                        results["dataset"] == self.dataset,
                    ),
                    results["date"] == datetime.today().strftime("%d %b %Y, %Hh"),
                )
            )[0]
            if len(ind) != 0:
                self.acc.append(results.iloc[ind[0]]["mean_objective"])
                self.std.append(results.iloc[ind[0]]["std_objective"])
                continue

            # Time & loops through Cross-Validation
            val_score = []
            train_score = []
            train_time = []
            for t, v in cross_val.split(X, Y):
                t_start = time.time()
                Xt, Xv, Yt, Yv = X[t], X[v], Y[t], Y[v]
                model = copy.copy(master_model)
                model.fit(Xt, Yt)
                val_score.append(self.scoring(model, Xv, Yv))
                train_score.append(self.scoring(model, Xt, Yt))
                train_time.append(time.time() - t_start)

            # Results
            results = results.append(
                {
                    "date": datetime.today().strftime("%d %b %Y"),
                    "model": type(model).__name__,
                    "dataset": self.dataset,
                    "params": model.get_params(),
                    "mean_objective": np.mean(val_score),
                    "std_objective": np.std(val_score),
                    "mean_time": np.mean(train_time),
                    "std_time": np.std(train_time),
                },
                ignore_index=True,
            )
            self.acc.append(np.mean(val_score))
            self.std.append(np.std(val_score))
            if self.store_models:
                joblib.dump(
                    model,
                    self.folder + type(model).__name__ + "_%.5f.joblib" % self.acc[-1],
                )
            if self.plot:
                plt.plot(p, alpha=0.2)
            print(
                "[Modelling] %s %s train/val: %.2f %.2f, training time: %.1f s"
                % (
                    type(model).__name__.ljust(60),
                    self.scoring._score_func.__name__,
                    np.mean(train_score),
                    np.mean(val_score),
                    time.time() - t_start,
                )
            )

        # Store CSV
        if self.store_results:
            results.to_csv(self.folder + "Initial_Models.csv")

        # Plot
        if self.plot:
            plt.plot(vo, c="k")
            if self.regression:
                ind = np.where(self.acc == np.min(self.acc))[0][0]
            else:
                ind = np.where(self.acc == np.max(self.acc))[0][0]
            p = self.models[ind].predict(vi)
            plt.plot(p, color="#2369ec")
            plt.title("Predictions")
            plt.legend(
                [
                    "True output",
                    "Ridge",
                    "Lasso",
                    "SGD",
                    "KNR",
                    "DTR",
                    "ADA",
                    "GBR",
                    "HGBR",
                    "MLP",
                    "Best",
                ]
            )
            plt.ylabel("Output")
            plt.xlabel("Samples")
            plt.show()
        return results


class Sequence(object):
    def __init__(self, back=0, forward=0, shift=0, diff="none"):
        """
        Sequencer. Sequences and differnces data.
        Scenarios:
        - Sequenced I/O                     --> back & forward in int or list of ints
        - Sequenced input, single output    --> back: int, forward: list of ints
        - Single input, single output       --> back & forward =

        :param back: Int or List[int]: input indices to include
        :param forward: Int or List[int]: output indices to include
        :param shift: Int: If there is a shift between input and output
        :param diff: differencing algo, pick between 'none', 'diff', 'logdiff', 'frac' (no revert)
        """
        if type(back) == int:
            back = np.linspace(0, back, back + 1).astype("int")
            self.inputDtype = "int"
        elif type(back) == list:
            back = np.array(back)
            self.inputDtype = "list"
        else:
            raise ValueError("Back needs to be int or list(int)")
        if type(forward) == int:
            self.outputDtype = "int"
            forward = np.linspace(0, forward, forward + 1).astype("int")
        elif type(forward) == list:
            self.outputDtype = "list"
            forward = np.array(forward)
        else:
            raise ValueError("Forward needs to be int or list(int)")
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
        if diff != "none":
            self.samples = 0
            self.nback -= 1
        if diff not in ["none", "diff", "logdiff"]:
            raise ValueError("Type should be in [None, diff, logdiff, frac]")

    def convert(self, X, Y, flat=False):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
            assert isinstance(Y, pd.DataFrame) or isinstance(
                Y, pd.core.series.Series
            ), "Input and Output need to be the same data type."
            return self.convert_pandas(X, Y)
        elif isinstance(X, np.ndarray):
            assert isinstance(
                Y, np.ndarray
            ), "Input and Output need to be the same data type."
            return self.convert_numpy(X, Y, flat=flat)
        else:
            TypeError(
                "Input & Output need to be same datatype, either Numpy or Pandas."
            )

    def convert_numpy(self, X, Y, flat=False):
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if Y.ndim == 1:
            Y = Y.reshape((-1, 1))
        samples = len(X) - self.mback - self.mfore - self.shift - 1
        features = len(X[0])
        input_sequence = np.zeros((samples, self.nback, features))
        output_sequence = np.zeros((samples, self.nfore))
        if self.diff == "none":
            for i in range(samples):
                input_sequence[i] = X[i + self.backVec]
                output_sequence[i] = Y[
                    i - 1 + self.mback + self.shift + self.foreVec
                ].reshape((-1))
            return input_sequence, output_sequence
        elif self.diff[-4:] == "diff":
            if self.diff == "logdiff":
                X = np.log(X)
                Y = np.log(Y)
            if (self.backVec == 0).all():
                self.backVec = np.array([0, 1])
            for i in range(samples):
                input_sequence[i] = X[i + self.backVec[1:]] - X[i + self.backVec[:-1]]
                output_sequence[i] = (
                    Y[i + self.mback + self.shift + self.foreVec]
                    - Y[i + self.mback + self.shift + self.foreRoll]
                ).reshape((-1))
            return input_sequence, output_sequence

    def convert_pandas(self, X, Y):

        # Check inputs
        if isinstance(X, pd.core.series.Series):
            X = X.to_frame()
        if isinstance(Y, pd.core.series.Series):
            Y = Y.to_frame()
        assert len(X) == len(Y)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(Y, pd.DataFrame)

        # Keys
        inputKeys = X.keys()
        outputKeys = Y.keys()

        # No Differencing
        if self.diff == "none":
            # Input
            for lag in self.backVec:
                keys = [key + "_" + str(lag) for key in inputKeys]
                X[keys] = X[inputKeys].shift(lag)

            # Output
            for shift in self.foreVec:
                keys = [key + "_" + str(shift) for key in outputKeys]
                Y[keys] = Y[YKeys].shift(-shift)

        # With differencing
        elif self.diff[-4:] == "diff":
            # Input
            for lag in self.backVec:
                # Shifted
                keys = [key + "_" + str(lag) for key in inputKeys]
                X[keys] = X[inputKeys].shift(lag)

                # Differenced
                dkeys = [key + "_d_" + str(lag) for key in inputKeys]
                X[dkeys] = X[inputKeys].shift(lag) - X[inputKeys]

            # Output
            for shift in self.foreVec:
                # Only differenced
                keys = [key + "_" + str(shift) for key in outputKeys]
                Y[keys] = Y[outputKeys].shift(lag) - Y[outputKeys]

        # Drop _0 (same as original)
        X = X.drop([key for key in X.keys() if "_0" in key], axis=1)
        Y = Y.drop([key for key in Y.keys() if "_0" in key], axis=1)

        # Return (first lags are NaN, last shifts are NaN
        return (
            X.iloc[lag : -shift if shift > 0 else None],
            Y.iloc[lag : -shift if shift > 0 else None],
        )

    def revert(self, differenced, original):
        # unsequenced integrating loop: d = np.hstack((d[0], d[0] + np.cumsum(dd)))
        if self.nfore == 1:
            differenced = differenced.reshape((-1, 1))
        Y = np.zeros_like(differenced)
        if self.diff == "none":
            return
        if self.diff == "logdiff":
            for i in range(self.nfore):
                Y[:, i] = (
                    np.log(original[self.mback + self.foreRoll[i] : -self.foreVec[i]])
                    + differenced[:, i]
                )
            return np.exp(Y)
        if self.diff == "diff":
            for i in range(self.nfore):
                Y[:, i] = (
                    original[self.mback + self.foreRoll[i] : -self.foreVec[i]]
                    + differenced[:, i]
                )
            return Y


class GridSearch(object):
    def __init__(self, model, params, cv=None, scoring=metrics.r2_score):
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

        if (
            scoring == metrics.mean_absolute_error
            or scoring == metrics.mean_squared_error
        ):
            self.best = [np.inf, 0]
        else:
            self.best = [-np.inf, 0]

    def _parse_params(self):
        k, v = zip(*self.params.items())
        self.parsed_params = [
            dict(zip(k, v)) for v in itertools.product(*self.params.values())
        ]
        print(
            "[GridSearch] %i folds with %i parameter combinations, %i runs."
            % (
                self.cv.n_splits,
                len(self.parsed_params),
                len(self.parsed_params) * self.cv.n_splits,
            )
        )

    def fit(self, X, Y):
        # Convert to Numpy
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
            X = np.array(X)
        if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.core.series.Series):
            Y = np.array(Y).reshape((-1))

        # Loop through parameters
        for i, param in tqdm(enumerate(self.parsed_params)):
            # print('[GridSearch] ', param)
            scoring = []
            timing = []
            for train_ind, val_ind in self.cv.split(X, Y):
                # Start Timer
                t = time.time()

                # Split data
                xtrain, xval = X[train_ind], X[val_ind]
                ytrain, yval = Y[train_ind], Y[val_ind]

                # Model training
                model = copy.copy(self.model)
                model.set_params(**param)
                model.fit(xtrain, ytrain)

                # Results
                scoring.append(self.scoring(model.predict(xval), yval))
                timing.append(time.time() - t)

            # Compare scores
            if scoring == mean_absolute_error or scoring == mean_squared_error:
                if np.mean(scoring) + np.std(scoring) <= self.best[0] + self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]
            else:
                if np.mean(scoring) - np.std(scoring) > self.best[0] - self.best[1]:
                    self.best = [np.mean(scoring), np.std(scoring)]

            # print('[GridSearch] [AutoML] Score: %.4f \u00B1 %.4f (in %.1f seconds) (Best score so-far: %.4f \u00B1 %.4f) (%i / %i)' %
            #       (datetime.now().strftime('%H:%M'), np.mean(scoring), np.std(scoring), np.mean(timing), self.best[0], self.best[1], i + 1, len(self.parsed_params)))
            self.result.append(
                {
                    "scoring": scoring,
                    "mean_objective": np.mean(scoring),
                    "std_objective": np.std(scoring),
                    "time": timing,
                    "mean_time": np.mean(timing),
                    "std_time": np.std(timing),
                    "params": param,
                }
            )
        return pd.DataFrame(self.result)


# Objects for Hyperparameter Optimization
class LSTM(object):
    def __init__(
        self,
        dense=[1],
        stacked=[100, 100],
        bidirectional=False,
        conv=False,
        conv_filters=64,
        conv_kernel_size=1,
        conv_pool_size=2,
        conv_seq=2,
        dropout=False,
        dropout_frac=0.1,
        regularization=1e-4,
        activation="tanh",
        loss="mse",
        lstm_input=None,
        optimizers=None,
        epochs=50,
        batch_size=256,
        shuffle=True,
        early_stopping=None,
    ):
        from keras import optimizers, regularizers
        from keras.layers import (
            LSTM,
            Bidirectional,
            Dense,
            Flatten,
            TimeDistributed,
            convolutional,
        )
        from keras.models import Sequential

        try:
            self.model = Sequential()
            if conv:
                i, j = lstm_input
                self.model.add(
                    TimeDistributed(
                        convolutional.Conv1D(
                            filters=conv_filters,
                            kernel_size=conv_kernel_size,
                            activation=activation,
                        ),
                        input_shape=conv_input,
                    )
                )
                self.model.add(
                    TimeDistributed(
                        convolutional.MaxPooling1D(
                            pool_size=(None, int(i / conv_seq), j)
                        )
                    )
                )
                self.model.add(TimeDistributed(Flatten()))
            if bidirectional:
                self.model.add(
                    Bidirectional(
                        LSTM(stacked[0], activation=activation), input_shape=lstm_input
                    )
                )
                for i in range(len(stacked) - 1):
                    self.model.add(LSTM(stacked[i + 1], activation=activation))
            else:
                self.model.add(
                    LSTM(stacked[0], activation=activation), input_shape=lstm_input
                )
                for i in range(len(stacked) - 1):
                    self.model.add(LSTM(stacked[i + 1], activation=activation))
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

    def fit(self, X, Y):
        self.model.fit(
            X,
            Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            callbacks=self.callbacks,
        )
        return self.model

    def predict(self, X):
        return self.model.predict(X)
