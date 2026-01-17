import numpy as np
from scipy.stats import genextreme

def ajustar_gev_maximos_anuais(serie_dados):
    """
    Ajusta a Distribuição Generalizada de Valores Extremos (GEV) 
    aos máximos anuais de uma série.
    
    Retorna:
        shape (c), loc, scale
    """
    # Agrupar por ano e pegar maximo
    df = serie_dados.copy()
    df['ano'] = df['data'].dt.year
    maximos_anuais = df.groupby('ano')['temperatura_max'].max()
    
    c, loc, scale = genextreme.fit(maximos_anuais)
    return c, loc, scale, maximos_anuais

def calcular_retorno_gev(c, loc, scale, anos_retorno=[10, 50, 100]):
    """
    Calcula valores de retorno para 10, 50, 100 anos.
    """
    valores = {}
    for t in anos_retorno:
        prob = 1 - 1/t
        valor = genextreme.ppf(prob, c, loc, scale)
        valores[t] = valor
    return valores
