import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def decompor_serie_temporal(serie, periodo=365):
    """
    Realiza decomposição sazonal (Tendência + Sazonalidade + Resíduo).
    Modelo aditivo.
    """
    resultado = seasonal_decompose(serie, model='additive', period=periodo, extrapolate_trend='freq')
    return resultado.trend, resultado.seasonal, resultado.resid
