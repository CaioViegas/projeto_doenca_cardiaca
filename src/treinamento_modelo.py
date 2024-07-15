import pandas as pd
from preprocessamento import preprocessamento_dados
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def avaliar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Evaluates a given machine learning model using accuracy, precision, recall, and F1-score metrics.

    Parameters:
    - modelo (sklearn.base.BaseEstimator): A trained machine learning model.
    - X_train (pandas.DataFrame): The training input data.
    - X_test (pandas.DataFrame): The testing input data.
    - y_train (pandas.Series): The training target variable.
    - y_test (pandas.Series): The testing target variable.

    Returns:
    A dictionary containing the evaluation metrics:
    - "Accuracy_train": The accuracy of the model on the training data.
    - "Accuracy_test": The accuracy of the model on the testing data.
    - "Precision_train": The precision of the model on the training data.
    - "Precision_test": The precision of the model on the testing data.
    - "Recall_train": The recall of the model on the training data.
    - "Recall_test": The recall of the model on the testing data.
    - "F1_train": The F1-score of the model on the training data.
    - "F1_test": The F1-score of the model on the testing data.
    """
    # Treinar o modelo com os dados de treino
    modelo.fit(X_train, y_train)

    # Prever os rótulos para os dados de treino e teste
    train_pred = modelo.predict(X_train)
    test_pred = modelo.predict(X_test)

    # Calcular métricas de avaliação para os dados de treino
    accuracy_train = accuracy_score(y_train, train_pred)
    precision_train = precision_score(y_train, train_pred, average='weighted')
    recall_train = recall_score(y_train, train_pred, average='weighted')
    f1_train = f1_score(y_train, train_pred, average='weighted')

    # Calcular métricas de avaliação para os dados de teste
    accuracy_test = accuracy_score(y_test, test_pred)
    precision_test = precision_score(y_test, test_pred, average='weighted')
    recall_test = recall_score(y_test, test_pred, average='weighted')
    f1_test = f1_score(y_test, test_pred, average='weighted')

    # Retornar um dicionário com as métricas calculadas
    return {
        "Accuracy_train": accuracy_train,
        "Accuracy_test": accuracy_test,
        "Precision_train": precision_train,
        "Precision_test": precision_test,
        "Recall_train": recall_train,
        "Recall_test": recall_test,
        "F1_train": f1_train,
        "F1_test": f1_test,
    }

def treinar_modelo(dataset, target):
    """
    Trains a set of machine learning models on the given dataset and target variable,
    and returns the model with the highest F1-score on the testing data.

    Parameters:
    - dataset (pandas.DataFrame): The input dataset containing the features and target variable.
    - target (str): The name of the target variable column in the dataset.

    Returns:
    - best_model (sklearn.base.BaseEstimator): The trained machine learning model with the highest F1-score on the testing data.
    """
    # Definição dos modelos a serem avaliados
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=4000),
        "GaussianNB": GaussianNB(),
        "GradientBoosting": GradientBoostingClassifier(random_state=101),
        "RandomForest": RandomForestClassifier(random_state=101),
        "DecisionTree": DecisionTreeClassifier(random_state=101),
        "XGBoost": XGBClassifier(random_state=101)
    }

    # Pré-processamento dos dados para obter X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = preprocessamento_dados(dataset, target)

    # Inicialização das variáveis para armazenar o melhor modelo e sua pontuação
    best_model = None
    best_score = 0

    # Iteração sobre cada modelo para avaliação
    for nome_modelo, modelo in modelos.items():
        # Avaliação do modelo
        resultados = avaliar_modelo(modelo, X_train, X_test, y_train, y_test)
        f1_test = resultados["F1_test"]
        
        # Impressão dos resultados de avaliação
        print(f'{nome_modelo} - Accuracy: {resultados["Accuracy_test"]}, Precision: {resultados["Precision_test"]}, Recall: {resultados["Recall_test"]}, F1: {f1_test}')

        # Atualização do melhor modelo com base no F1-score
        if f1_test > best_score:
            best_score = f1_test
            best_model = modelo

            # Impressão do melhor modelo até o momento
            print(f'\nMelhor Modelo: {best_model.__class__.__name__} com F1-Score: {best_score}')

    # Retorno do melhor modelo encontrado
    return best_model

if __name__ == "__main__":
    df = pd.read_csv("data/heart.csv")
    treinar_modelo(df, 'HeartDisease')