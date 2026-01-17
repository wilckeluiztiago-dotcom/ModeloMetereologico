import numpy as np
import pandas as pd

def calcular_autocorrelacao(serie, lag=30):
    """
    Calcula a função de autocorrelação (ACF) até o lag especificado.
    """
    acf_valores = []
    for i in range(lag + 1):
        coef = serie.autocorr(lag=i)
        acf_valores.append(coef)
    return acf_valores
