import numpy as np
import pandas as pd

def gerar_dados_sc(data_inicio, data_fim):
    """
    Gera dados climáticos sintéticos realistas para Santa Catarina.
    Características: Litoral úmido, Serra fria. Média ponderada.
    """
    datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    n = len(datas)
    dias_do_ano = datas.dayofyear
    
    # SC: Média ~19-21C
    temp_media_sazonal = 20 + 6 * np.sin((2 * np.pi * (dias_do_ano - 280)) / 365)
    ruido = np.random.normal(0, 3.0, n)
    tendencia = np.linspace(0, 0.7, n)
    
    temperatura_media = temp_media_sazonal + ruido + tendencia
    temperatura_max = temperatura_media + np.random.uniform(4, 10, n)
    temperatura_min = temperatura_media - np.random.uniform(3, 8, n)
    
    # Chuva em SC é bem distribuída
    prob_chuva = 0.40
    chuva_ocorr = np.random.rand(n) < prob_chuva
    chuva_qtd = np.zeros(n)
    chuva_qtd[chuva_ocorr] = np.random.gamma(shape=1.8, scale=12, size=np.sum(chuva_ocorr))
    
    df = pd.DataFrame({
        'data': datas,
        'temperatura_max': temperatura_max,
        'temperatura_min': temperatura_min,
        'precipitacao': chuva_qtd,
        'umidade': np.clip(80 + np.random.normal(0, 8, n), 40, 100),
        'pressao': 1015 + np.random.normal(0, 3, n),
        'vento_vel': np.abs(np.random.weibull(1.8, n) * 4.5),
        'estado': 'SC'
    })
    return df
