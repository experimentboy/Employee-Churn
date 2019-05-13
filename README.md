# Employee-Churn
Data Science with Employee Churn related dataset

After reading this article (https://towardsdatascience.com/building-an-employee-churn-model-in-python-to-develop-a-strategic-retention-plan-57d5bd882c2d), i was using the same dataset to verify potential added values of my approach.


## Results of the experiment:

Using TPOT² algorithm, the best prediction using traditional (statistical) models is arround 0.87.
This CV score is obtained using this pipeline:

Best pipeline: ExtraTreesClassifier(input_matrix, bootstrap=False, criterion=entropy, max_features=0.55, min_samples_leaf=13, min_samples_split=2, n_estimators=100)

TPOTClassifier(config_dict=None, crossover_rate=0.1, cv=5,
        disable_update_check=False, early_stop=5, generations=5,
        max_eval_time_mins=5, max_time_mins=None, memory=None,
        mutation_rate=0.9, n_jobs=-1, offspring_size=None,
        periodic_checkpoint_folder=None, population_size=20,
        random_state=42, scoring=None, subsample=1.0,
        template='RandomTree', use_dask=False, verbosity=2,
        warm_start=False)

²: TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

Using deep learning approach and SMOTE³, my standard tabular leaner (FastAI) demonstrate 
supremacy at the 2nd epoch with perfect accuracy score (1.0).

³SMOTE - Synthetic Minority Over-sampling Technique as presented in https://imbalanced-learn.org/

ALl 3 kernels of this experiment can run on CPU (GPU is faster for part 3). 

## Related work:

Part 1 : Can we identify most important features using LOFO ?

Part 2 : What would be the the best traditional modeling approach ?

Part 3 : Can we predict employee churn uing deep learning ?

The kernels published here and related dataset are also available on Kaggle (https://www.kaggle.com/sambapython/thirdday/kernels).
