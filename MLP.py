#!/usr/bin/env python
# coding: utf-8

# Import libraries
import ast
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import jaccard_score, f1_score, precision_score 
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score, fbeta_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

def encode_variables(df, X_names_list, Y_name):
    """ This function scale and encodes the features and target variables.
    df: the dataframe to be processed. Dtype: pandas.core.frame.DataFrame
    X_names_list: list of feature(s) name. Dtype: list or str
    Y_name: list of target variable(s) name. Dtype: list or str
    """

    # Create dataframe for features
    X=pd.DataFrame(df.loc[:,X_names_list])

    # Get values of target variables
    Y=df.loc[:,Y_name].values

    # Find the categorical columns
    categorical_cols = df[X_names_list].select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df[X_names_list].select_dtypes(include=['int64','float64']).columns.tolist()

    # Initialize the OneHotEncoder and MinMaxScaler
    onehot_encoder = OneHotEncoder()
    scaler = MinMaxScaler()

    # Encode the categorical columns
    encoded_categorical_cols = onehot_encoder.fit_transform(df[categorical_cols])

    # Scale the numeric columns
    scaled_numeric_cols = scaler.fit_transform(df[numeric_cols])

    # Create a new Dataframe with the encoded and scaled columns
    encoded_df = pd.DataFrame(encoded_categorical_cols.toarray(),
                    columns=onehot_encoder.get_feature_names(categorical_cols))
    scaled_df = pd.DataFrame(scaled_numeric_cols, columns=numeric_cols)

    # Concat the encoded and scaled Dataframe with the original Dataframe
    X = pd.concat([encoded_df, scaled_df], axis=1)

    return X, Y

def read_config(model_):
    """ This function reads the 'config.csv' file and process the parameters.
    model_: name of the model. Dtype: str
    """

    # Read config file and slice
    df = pd.read_csv('config.csv',index_col=0)
    config = pd.DataFrame(df.loc[:,model_]).dropna().to_dict().pop(model_)

    # Loop over the dictionary and convert the values to the desired types
    for key, value in config.items():
        # Convert strings that represent integers or floats to their corresponding types
        if value.isdigit():
            config[key] = int(value)
        elif value.replace('.', '', 1).isdigit():
            config[key] = float(value)
        # Convert strings that represent booleans to their corresponding type
        elif value in ['TRUE', 'FALSE', 'None']:
            value = value.capitalize()
            config[key] = ast.literal_eval(value)
        # Remove single quotation marks from string values
        elif value.startswith('"') and value.endswith('"'):
            config[key] = value[1:-1]

    return config

def format_perc(x_):
    """ This function converts float value to percentage in 2 decimal places.
    x_: value to convert. Dtype: float
    """
    return f'{x_:.2%}'

def ML_pipeline(split_ratio, rand_state, df, X_names_list, Y_name):
    """ This function begins the machine learning pipline and output the evaluation scores.
    split_ratio: split ratio for train-test split. Dtype: float
    rand_state: random seed number for train-test split. Dtype: int
    df: dataframe of feature(s) and target variable(s). Dtype: pandas.core.frame.DataFrame
    X_names_list: list of feature(s) name. Dtype: list or str
    Y_name: list of target variable(s) name. Dtype: list or str
    """

    # Encode features and target variables
    X, Y = encode_variables(df=df, X_names_list=X_names_list, Y_name=Y_name)

    # Dividing total sets of data into test set and training set
    X_train, X_test, y_train, y_test=train_test_split(X,
                                                    Y,
                                                    test_size=split_ratio,
                                                    random_state=rand_state)

    # Read config file, process the values and convert it to dictionary of dictionaries
    configs = {model: read_config(model) for model in pd.read_csv('config.csv',
                                                                index_col=0).columns.to_list()}

    # Creating instances of machine learning piplines
    DT_CC_Pipeline=Pipeline([('classifier',
                        ClassifierChain(DecisionTreeClassifier(**configs['DecisionTreeClassifier'])))])

    AB_CC_Pipeline=Pipeline([('classifier',
                        ClassifierChain(AdaBoostClassifier(n_estimators=1000)))])

    RF_CC_Pipeline=Pipeline([('classifier',
                        ClassifierChain(RandomForestClassifier(**configs['RandomForestClassifier'])))])

    NN_Pipeline=Pipeline([('classifier',
                        MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100),
                        max_iter=500,
                        random_state=42))])

    # defining the pipelines in a list
    mypipeline = [DT_CC_Pipeline, AB_CC_Pipeline, RF_CC_Pipeline, NN_Pipeline]

    # Defining and initializing variables for evaluating the best model
    results_df = pd.DataFrame({'Model': [],
        'Precision': [],
        'Recall': [],
        'F1_Weighted': [],
        'ROC AUC': [],
        'Jaccard Similarity': [],
        'Precision-Recall Curve': [],
        'F Beta': []})

    # Creating dictionary of pipelines and training models
    PipelineDict = {0: 'DT_CC', 1: 'AB_CC', 2: 'RF_CC', 3: 'NN'}

    # Fit the pipelines
    for mypipe in mypipeline:
        mypipe.fit(X_train, y_train)

    # Getting evaluation metrics for all models, one at a time
    for i,model in enumerate(mypipeline):
        condition_MLP = model.named_steps['classifier'].__class__.__name__ == 'MLPClassifier'
        condition_RF = model.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'
        if (condition_MLP) | (condition_RF):
            ROC_ = roc_auc_score(y_test, model.predict(X_test), **configs['ROC'])
            f1_weighted, prec_, rec_, jac_, PRC_, FB_ = eva_metrics(y_pred=model.predict(X_test),
                                                                        y_pred_arr=model.predict(X_test),
                                                                        y_test=y_test,
                                                                        configs=configs)
        else:
            ROC_ = roc_auc_score(y_test, model.predict(X_test).toarray(), **configs['ROC'])
            f1_weighted, prec_, rec_, jac_, PRC_, FB_ = eva_metrics(y_pred=model.predict(X_test),
                                                                        y_pred_arr=model.predict(X_test).toarray(),
                                                                        y_test=y_test, configs=configs)

        # Consolidating evaluation metrics 
        new_row = {'Model':PipelineDict[i], 'F1_Weighted':f1_weighted, 'Precision':prec_,
                    'Recall':rec_, 'ROC AUC':ROC_, 'Jaccard Similarity':jac_,
                    'Precision-Recall Curve':PRC_, 'F Beta':FB_}
        results_df = results_df.append(new_row, ignore_index=True)

    list_seq = ['F1_Weighted', 'Precision', 'Recall',
                'ROC AUC', 'Jaccard Similarity', 'Precision-Recall Curve',
                'F Beta']

    # Format result dataframe to percentage, in 2 decimal places
    results_df[list_seq] = results_df[list_seq].applymap(format_perc)
    results_df.index += 1
    pd.options.display.max_rows = None
    print('\nEvaluation metrics:\n',tabulate(results_df, headers='keys',
                                            tablefmt='psql', stralign="center"))

    return results_df

def eva_metrics(y_pred, y_test, y_pred_arr, configs):
    """ This function calculates the different evaluation metrics.
    y_pred: predicted result. Dtype: scipy.sparse._lil.lil_matrix or numpy.ndarray
    y_test: test set of target variable(s). Dtype: int
    y_pred_arr: predicted result in array format. Dtype: numpy.ndarray
    configs: configuration parameters of metrics. Dtype: dict
    """

    f1_weighted = f1_score(y_pred,y_test, **configs['F1_Weighted'])
    prec_ = precision_score(y_pred,y_test, **configs['Precision'])
    rec_ = recall_score(y_pred,y_test, **configs['Recall'])
    jac_ = jaccard_score(y_pred_arr,y_test, **configs['Jaccard'])
    PRC_ = average_precision_score(y_pred_arr,y_test, **configs['Precision_Recall'])
    FB_ = fbeta_score(y_pred,y_test, **configs['F_Beta'])

    return f1_weighted, prec_, rec_, jac_, PRC_, FB_