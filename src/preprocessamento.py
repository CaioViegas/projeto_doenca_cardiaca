import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def preprocessamento_dados(dataset, target, test_size=0.2, random_state=101, pca_components=10, k_best=10):
    """
    Preprocessamento de dados para um modelo de classificação de doenças cardíacas.
    Realiza etapas de limpeza, codificação, normalização, redução de dimensionalidade e seleção de features.

    Parameters:
    dataset (pandas.DataFrame): O conjunto de dados de entrada contendo as features e a variável alvo.
    target (str): O nome da coluna que representa a variável alvo.
    test_size (float, optional): A proporção do conjunto de dados a ser reservada para teste. Default é 0.2.
    random_state (int, optional): O estado aleatório para reproduzir resultados consistentes. Default é 101.
    pca_components (int, optional): O número de componentes principais a serem mantidos após a redução de dimensionalidade PCA. Default é 5.
    k_best (int, optional): O número de melhores features a serem selecionadas após a seleção de features SelectKBest. Default é 5.

    Returns:
    tuple: Uma tupla contendo os conjuntos de dados de treino e teste pré-processados, respectivamente.
    """
    # Filtrar linhas onde 'Cholesterol' é diferente de 0
    dataset = dataset[dataset['Cholesterol'] != 0]

    # Codificar variáveis categóricas usando LabelEncoder
    le = LabelEncoder()
    colunas_label = ['Sex', 'ExerciseAngina']
    for col in colunas_label:
        dataset.loc[:, col] = le.fit_transform(dataset[col])

    # Realizar one-hot encoding em colunas categóricas
    colunas_ohe = ['ChestPainType', 'RestingECG', 'ST_Slope']
    ohe = OneHotEncoder(sparse_output=False)
    dataset_enc = pd.DataFrame(ohe.fit_transform(dataset[colunas_ohe]), columns=ohe.get_feature_names_out(), index=dataset.index)
    dataset_enc = pd.concat([dataset.drop(colunas_ohe, axis=1), dataset_enc], axis=1)

    # Salvar o dataset codificado em um arquivo CSV
    dataset_enc.to_csv("data/dataset_codificado.csv", index=False)

    # Preparar X (atributos) e y (variável alvo) para modelagem
    X = dataset_enc.drop(columns=[target])
    y = dataset_enc[target]

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Escalonar os atributos usando RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Realizar PCA para redução de dimensionalidade
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Selecionar as melhores k features usando SelectKBest com F-value do ANOVA
    seletor = SelectKBest(score_func=f_classif, k=k_best)
    X_train_selecionado = seletor.fit_transform(X_train_pca, y_train)
    X_test_selecionado = seletor.transform(X_test_pca)

    # Retornar os dados processados
    return X_train_selecionado, X_test_selecionado, y_train, y_test

if __name__ == '__main__':
    df = pd.read_csv("data/heart.csv")
    preprocessamento_dados(df)
