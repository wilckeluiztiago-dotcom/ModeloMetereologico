import os
import pandas as pd
from nucleo.configuracao import DIR_TABELAS

def salvar_resultados_csv(df, nome_arquivo):
    caminho = os.path.join(DIR_TABELAS, f"{nome_arquivo}.csv")
    df.to_csv(caminho, index=False)
    print(f"Salvo: {caminho}")

def salvar_relatorio_texto(texto, nome_arquivo):
    caminho = os.path.join(DIR_TABELAS, f"{nome_arquivo}.txt")
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"Relatório salvo: {caminho}")
