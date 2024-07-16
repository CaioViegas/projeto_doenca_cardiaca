import pandas as pd
from dataset_info import dataset_informacao
from otimizacao import modelo_otimizado

if __name__ == '__main__':
    df = pd.read_csv("data/House Price India.csv")
    dataset_informacao(df)
    modelo_otimizado(df, 'Price')