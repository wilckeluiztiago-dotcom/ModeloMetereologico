import pandas as pd
import numpy as np

def limpar_outliers_temperatura(df):
    """
    Remove valores físicos impossíveis para o Sul do Brasil.
    Temp > 50C ou < -15C são considerados erros de medição/geração.
    """
    df_clean = df.copy()
    mask = (df_clean['temperatura_max'] > 50) | (df_clean['temperatura_min'] < -15)
    
    # Interpolação linear para preencher buracos removidos
    df_clean.loc[mask, ['temperatura_max', 'temperatura_min']] = np.nan
    df_clean.interpolate(method='linear', inplace=True)
    
    return df_clean
