import os
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier

class RFEH:
    """ Feature selection and hyperparameters tuning for a classifier.
    
    Uses the Recursive Feature Elimination method (RFE) and Grid Search. 

    For each feature set, it obtains the best hyperparameters and scores 
    the best model for the subset. After finding the best model for each 
    feature set, it returns the one with the best results along with the 
    best hyperparameters for it.
    
    Attributes
    ----------
    feature_importances_ : dict
        Features associated to their importance score.

    results_summary_ : dict
        Summary of the results, consider the number of features, score in
        every subset and the best hyperparameters for this one.
        
    best_result_ : dict
        Number of features, hyperparameters and score of best result.

    full_results_ : pandas Dataframe
        Grid search results for every subset of features. Is a collection 
        of 'cv_results_' dictionaries from grid search results.
    """

    def __init__(self, train_data, val_data, model, params_grid,
                 metrics, rank_score = 'gini', seed = None):
        """
        Parameters
        ----------
        train_data : list[pandas DataFrame]
            Train data. Expect a list like: [X_train, y_train]

        val_data : list[pandas DataFrame]
            Validation data. Expect a list like: [X_val, y_val]

        model : sklearn ``Estimator`` instance
            A supervised learning estimator with a ``fit`` method.

        params_grid : dict
            The hyperparameters to explore, as a dictionary mapping 
            estimator parameters to sequences of allowed values.

            An empty dict signifies default parameters.

        metrics : str or list
            Measure to evaluate the performance of the model to 
            select the best configuration.

        rank_score : str or function, optional, default = 'gini'
            Measure to rank the feature importances. For default, the
            gini importance is calculated, with a random forest model. 

            Can receive a function to calculate the feature importances
            with other measure, as an statistical method. 

        seed : int, optional, default = None
            Number for reproducibility. If is None, the experiment
            will be different each time than runs.
        """
        self.X_train     = train_data[0]
        self.y_train     = train_data[1]
        self.X_val       = val_data[0]
        self.y_val       = val_data[1]
        self.model       = model
        self.params_grid = params_grid
        self.metrics     = metrics
        self.rank_score  = rank_score
        self.seed        = seed

    def rank_features(self):
        """Sort the features in increasing order by a given rating measure.

        By default, if there is no given measure uses the gini importance.

        Returns
        -------
        list
            Sorted features names, starting with the less important to 
            remove first in the RFE feature selection.
        """

        if self.rank_score == 'gini':
            ranked_features = self.gini_importance()
        else:
            ranked_features = {}
            for col in self.X_train.columns:
                ranked_features[col] = self.rank_score(self.X_train[col])

        self.feature_importances_ = ranked_features
        return sorted(ranked_features)
    
    def gini_importance(self):
        """Calculate the gini importance for rank features.

        Returns
        -------
        dict
            Features associated to their gini importance.
        """
        rf = RandomForestClassifier(random_state=self.seed)
        rf.fit(self.X_train, self.y_train)
        importances = rf.feature_importances_
        gini_importances = pd.Series(importances, index=self.X_train.columns)
        return dict(gini_importances)

    def hyperparams_tuning(self, X, y, predefined_split):
        """Search for the best hyperparameters for a subset of features using grid search.

        Parameters
        ----------
        X : numpy ndarray
            Train and validation data concatened in an array.

        y : numpy ndarray
            Classes of the train and validation data concatened in an array.

        predefined_split : sklearn ``PredefinedSplit`` instance
            Train/validation indices to split data in a predefined scheme. 
            Grid search will use this indices to define the data in cross-validation.

        Returns
        -------
        dict of numpy (masked) ndarrays
            A dict with results of the grid search.
        dict
            Parameter setting that gave the best results on the hold out data.
        float
            Mean cross-validated score of the best_estimator
        """
        grid = GridSearchCV(
                estimator  = self.model,
                param_grid = self.params_grid,
                scoring    = self.metrics,
                cv         = predefined_split,
                n_jobs     = multiprocessing.cpu_count() - 1,
                )
        grid.fit(X, y)
        return grid.cv_results_, grid.best_params_, grid.best_score_

    def run(self, n_reload = None, save_temp = False, save_path = '.'):
        """Run the feature selection and hyperparameters tuning with RFE

        Parameters
        ----------
        n_reload : int, optional, default = None
            Number of features to reload the selection and hyperparams search.

        save_temp : bool, optional, default = False
            True to save csv files with the results of the grid search for every set 
            of features, to reload later or like a backup.
            
        save_path : str, optional, default = '.'
            Path where will be saved the temporal results in csv files.
        
        Returns
        -------
        dict
            Number of features, hyperparameters and score of best result.
        """

        # Get the features sorted by importance
        ranked_features = self.rank_features()

        # Start selection from a given number of features
        n_reload = len(ranked_features) if n_reload == None else n_reload

        self.results_summary_ = {
                                    'n_features' : [],
                                    'hyperparams': [],
                                    'score'      : []
                                }
        
        for start in range(len(ranked_features)-n_reload, len(ranked_features)):
            # The less important features are removed in every iteration
            # 'start' index define the start of next subset of features
            current_features = ranked_features[start:]

            # Define train and validation splits indices for grid search
            X_train_validation = np.concatenate((self.X_train[current_features], 
                                                 self.X_val[current_features]), axis = 0)
            y_train_validation = np.concatenate((self.y_train, self.y_val), axis = 0)

            train_index = (-1*np.ones(len(self.X_train[current_features])))
            validation_index = (0*np.ones(len(self.X_val[current_features])))

            split_index = np.concatenate((train_index, validation_index), axis = 0)
            ps = PredefinedSplit(test_fold = split_index)

            # Grid Search -------------------------------------------------------------------------
            results, best_params, best_score = self.hyperparams_tuning(
                    X = X_train_validation, 
                    y = y_train_validation,
                    predefined_split = ps
                )
            # Save grid search results for actual set of features ---------------------------------
            n_features = len(ranked_features)-start
            n_results = len(list(results.values())[0])

            if save_temp: 
                try: os.makedirs(f"./{save_path}") 
                except: 
                    pass
                df = pd.DataFrame(results)
                df.to_csv(f'{save_path}/{n_features}_features.csv')
            
            # Save the results for every set of features ------------------------------------------
            if start == len(ranked_features)-n_reload and n_reload != len(ranked_features):
                reload_data = self.reload_results(save_path = save_path, 
                                                  n_reload = n_reload, 
                                                  n_features = len(ranked_features)
                                                  )
                search_results, self.results_summary_ = reload_data

            elif start == len(ranked_features)-n_reload and n_reload == len(ranked_features):
                search_results = pd.DataFrame(results)
                search_results['n_features'] = [n_features]*n_results
            else:
                results['n_features'] = [n_features]*n_results
                results_df = pd.DataFrame(results)
                search_results = pd.concat([search_results, results_df])

            self.results_summary_['score'].append(best_score)
            self.results_summary_['hyperparams'].append(best_params)
            self.results_summary_['n_features'].append(n_features)

        # Summary results -------------------------------------------------------------------------
        df_results = pd.DataFrame(self.results_summary_)
        best_result = df_results.sort_values('score', ascending = False).iloc[0]

        self.full_results_ = search_results.set_index('n_features')
        self.best_result_ = best_result.to_dict()
        
        return self.best_result_
    
    def reload_results(self, save_path, n_reload, n_features):
        """Reload previous results of features selection.

        Returns all the results so far and the summary
        of the best results for every set of features.

        Parameters
        ----------
        save_path : str
            Path where are the previous results.
        n_reload : int
            Number of features to reload the selection and hyperparams search.
        n_features : int
            Number of total features in the dataset.

        Returns
        -------
        pandas DataFrame
            Collection of the previous results.
        dict
            Results summary of previous results.

        Raises
        ------
        Exception
            If the csv files are not found, there are no previous results so 
            it cannot be reloaded.
        """
        try:
            cols = list(pd.read_csv(f'{save_path}/{n_reload+1}_features.csv').columns)
            cols.append('n_features')
            previous_results = pd.DataFrame(columns=cols)
            best_results = {
                                'n_features' : [],
                                'hyperparams': [],
                                'score'      : []
                            }
            for i in range(n_reload+1, n_features+1):
                temp = pd.read_csv(f'{save_path}/{i}_features.csv')
                temp['n_features'] = [i]*temp.shape[0]
                best_result = temp.sort_values('mean_test_score', ascending = False).iloc[0]
                best_results['n_features'].append(i)
                best_results['hyperparams'].append(best_result['params'])
                best_results['score'].append(best_result['mean_test_score'])
                previous_results = pd.concat([previous_results, temp])
            
            previous_results = previous_results.sort_values('n_features', ascending = False)
            previous_results = previous_results.set_index('n_features')
            previous_results = previous_results.drop(['Unnamed: 0'], axis=1)
            return previous_results, best_results
        
        except:
            raise Exception(f"(!) Previous results not found. Cannot reload.")