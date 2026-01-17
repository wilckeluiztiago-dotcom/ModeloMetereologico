import pandas as pd
import numpy as np

def calcular_spi_simplificado(precipitacao, janela=30):
    """
    Índice de Precipitação Padronizado (SPI - Standardized Precipitation Index) Simplificado.
    Calcula anomalias padronizadas da precipitação acumulada móvel.
    """
    precip_acumulada = precipitacao.rolling(window=janela).sum()
    
    media = precip_acumulada.mean()
    desvio = precip_acumulada.std()
    
    spi = (precip_acumulada - media) / desvio
    
    # SPI < -1.5 indica seca severa
    dias_seca = np.sum(spi < -1.5)
    
    return spi, dias_seca
