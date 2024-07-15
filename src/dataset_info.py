import pandas as pd

def informacao_dataset(dataset):
    """
    This function provides a comprehensive analysis of a given pandas DataFrame,
    including the first few rows, column information, statistical summary, null values,
    and distribution of categorical variables.

    Parameters:
    dataset (pandas.DataFrame): The dataset to be analyzed.

    Returns:
    None. The function prints the analysis results directly to the console.
    """
    print("Primeiras linhas do dataset:")
    print(dataset.head())

    print("\nInformações das colunas do dataset:")
    print(dataset.info())

    print("\nResumo estatístico das colunas:")
    print(dataset.describe())

    print("\nValores nulos:")
    print(dataset.isnull().sum())

    colunas_obj = [col for col in dataset.columns if dataset[col].dtype == 'object']

    for col in colunas_obj:
        print(f"\nDistribuição de valores em '{col}':")
        print(dataset[col].value_counts())

    colunas_num = [col for col in dataset.columns if dataset[col].dtype in ['int64', 'float64']]
    
    print("\nQuantidade de outliers em cada coluna numérica:")
    for col in colunas_num:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = dataset[(dataset[col] < (Q1 - 1.5 * IQR)) | (dataset[col] > (Q3 + 1.5 * IQR))]
        print(f"{col}: {outliers.shape[0]}")

if __name__ == '__main__':
    df = pd.read_csv("data/heart.csv")
    informacao_dataset(df)