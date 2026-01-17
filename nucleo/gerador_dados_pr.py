import numpy as np
import pandas as pd

def gerar_dados_pr(data_inicio, data_fim):
    """
    Gera dados climáticos sintéticos realistas para o Paraná.
    Características: Norte mais quente, Sul mais frio.
    """
    datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    n = len(datas)
    dias_do_ano = datas.dayofyear
    
    # PR: Média ~20-22C
    temp_media_sazonal = 21 + 5.5 * np.sin((2 * np.pi * (dias_do_ano - 280)) / 365)
    ruido = np.random.normal(0, 3.2, n)
    tendencia = np.linspace(0, 0.9, n) # Aquecimento um pouco maior no norte
    
    temperatura_media = temp_media_sazonal + ruido + tendencia
    # Maior amplitude térmica em algumas regiões
    temperatura_max = temperatura_media + np.random.uniform(6, 13, n)
    temperatura_min = temperatura_media - np.random.uniform(5, 9, n)
    
    prob_chuva = 0.38
    chuva_ocorr = np.random.rand(n) < prob_chuva
    chuva_qtd = np.zeros(n)
    chuva_qtd[chuva_ocorr] = np.random.gamma(shape=2.2, scale=11, size=np.sum(chuva_ocorr))
    
    df = pd.DataFrame({
        'data': datas,
        'temperatura_max': temperatura_max,
        'temperatura_min': temperatura_min,
        'precipitacao': chuva_qtd,
        'umidade': np.clip(78 + np.random.normal(0, 9, n), 35, 100),
        'pressao': 1012 + np.random.normal(0, 3.5, n),
        'vento_vel': np.abs(np.random.weibull(2.1, n) * 4.8),
        'estado': 'PR'
    })
    return df
