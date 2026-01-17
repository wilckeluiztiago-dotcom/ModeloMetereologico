import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def gerar_dados_rs(data_inicio, data_fim):
    """
    Gera dados climáticos sintéticos realistas para o Rio Grande do Sul.
    Características: Verão quente, inverno frio, chuvas bem distribuídas mas com variabilidade.
    Influência forte de frentes frias.
    """
    datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    n = len(datas)
    
    # Sazonalidade anual para temperatura
    dias_do_ano = datas.dayofyear
    # RS: Média anual ~18-20C, Amplitude alta
    temp_media_sazonal = 19 + 8 * np.sin((2 * np.pi * (dias_do_ano - 280)) / 365)
    
    # Ruído diário (frentes frias, variações)
    ruido_temp = np.random.normal(0, 3.5, n)
    
    # Tendência de aquecimento (Mudança climática 1990-2024: ~ +0.8C)
    tendencia = np.linspace(0, 0.8, n)
    
    temperatura_media = temp_media_sazonal + ruido_temp + tendencia
    temperatura_maxima = temperatura_media + np.random.uniform(5, 12, n)
    temperatura_minima = temperatura_media - np.random.uniform(4, 10, n)
    
    # Precipitação: Mais frequente no inverno/primavera, mas presente o ano todo
    # Modelo de chuva: processo de Poisson composto ou similar
    prob_chuva = 0.35 + 0.1 * np.sin((2 * np.pi * (dias_do_ano)) / 365) # Mais chance no inverno
    chuva_ocorrencia = np.random.rand(n) < prob_chuva
    chuva_quantidade = np.zeros(n)
    # Chuvas variam de garoa a tempestades (distribuição gama)
    chuva_quantidade[chuva_ocorrencia] = np.random.gamma(shape=2, scale=10, size=np.sum(chuva_ocorrencia))
    
    # El Niño (simplificado) - Aumenta chuva no RS
    # Ciclo de ~3-7 anos
    anos = datas.year
    indice_enso = np.sin((2 * np.pi * anos) / 5) 
    # Quando ENSO > 0.5 (El Niño), chuva aumenta
    mask_el_nino = indice_enso > 0.5
    chuva_quantidade[mask_el_nino] *= 1.4
    
    df = pd.DataFrame({
        'data': datas,
        'temperatura_max': temperatura_maxima,
        'temperatura_min': temperatura_minima,
        'precipitacao': chuva_quantidade,
        'umidade': np.clip(75 + 10 * np.cos((2 * np.pi * dias_do_ano)/365) + np.random.normal(0, 10, n), 30, 100),
        'pressao': 1013 - 5 * np.sin((2 * np.pi * dias_do_ano)/365) + np.random.normal(0, 4, n),
        'vento_vel': np.abs(np.random.weibull(2, n) * 5),
        'estado': 'RS'
    })
    
    return df
