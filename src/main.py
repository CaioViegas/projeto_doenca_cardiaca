import pandas as pd
from dataset_info import informacao_dataset
from otimizacao import modelo_otimizado

if __name__ == '__main__':
    df = pd.read_csv("data/heart.csv")
    informacao_dataset(df)
    modelo_otimizado(df, 'HeartDisease')