import pandas as pd

def analisar_risco_geada(df):
    """
    Identifica dias com risco de geada no Sul.
    Critério: Min Temp < 3°C (Geada Fraca), < 0°C (Geada Forte)
    """
    geada_fraca = df[df['temperatura_min'] < 3]
    geada_forte = df[df['temperatura_min'] < 0]
    
    resumo = {
        'dias_geada_fraca': len(geada_fraca),
        'dias_geada_forte': len(geada_forte),
        'probabilidade_anual': len(geada_fraca) / (len(df)/365.0)
    }
    return resumo
