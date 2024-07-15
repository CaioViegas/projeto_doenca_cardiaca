import pandas as pd
import joblib
from preprocessamento import preprocessamento_dados
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV

def modelo_otimizado(dataset, target):
    """
    This function optimizes a Logistic Regression model using GridSearchCV.
    It performs hyperparameter tuning and cross-validation to find the best parameters for the model.

    Parameters:
    - dataset (pandas DataFrame): The input dataset containing the features and target variable.
    - target (str): The target variable column name in the dataset.

    Returns:
    - None: The function prints the best parameters found, the accuracy, precision, recall, and F1-score of the model.
    """
    # Criando uma instância do modelo de regressão logística
    log = LogisticRegression(max_iter=2000)

    # Realizando o pré-processamento dos dados
    X_train, X_test, y_train, y_test = preprocessamento_dados(dataset, target)

    # Definindo o grid de hiperparâmetros para busca
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l2'],
        'max_iter': [2000, 4000, 6000],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'multi_class': ['ovr'],
        'tol': [0.001, 0.0001, 0.00001]
    }

    # Criando um objeto GridSearchCV para otimização
    grid_search = GridSearchCV(estimator=log, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

    # Realizando a busca de hiperparâmetros
    grid_search.fit(X_train, y_train)
    
    # Obtendo o melhor modelo encontrado
    best_model = grid_search.best_estimator_ 

    # Imprimindo os melhores parâmetros encontrados
    print(f'Melhores parâmetros encontrados:')
    print(best_model.get_params())
    
    # Realizando previsões no conjunto de teste
    y_pred = best_model.predict(X_test)
    
    # Calculando métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Imprimindo o melhor modelo
    print(best_model)
    
    # Realizando validação cruzada para avaliar a acurácia
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross-Validation Accuracy: {cross_val_scores.mean():.2f}')
    print(f'Acurácia: {accuracy:.2f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')

    # Salvar o modelo
    joblib.dump(best_model, 'modelo_escolhido.joblib')

    return best_model

if __name__ == '__main__':
    df = pd.read_csv("data/heart.csv")
    modelo_otimizado(df, 'HeartDisease')