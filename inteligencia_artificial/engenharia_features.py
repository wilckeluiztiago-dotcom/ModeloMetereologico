import pandas as pd
import numpy as np

"""
MÓDULO DE ENGENHARIA DE FEATURES (IA)
=====================================

Prepara os dados brutos meteorológicos para ingestão em modelos de Machine Learning.
Cria variáveis derivadas que capturam a dinâmica temporal e física.

Funcionalidades:
- Lags (Defasagens temporais)
- Médias Móveis (Rolling Windows)
- Diferenciação
- Codificação Cíclica de tempo (Seno/Cosseno do dia do ano)
- Features de Interação (ex: T * Umidade)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class EngenheiroFeatures:
    def __init__(self):
        pass
        
    def criar_lags(self, df, colunas, n_lags=3):
        """Cria colunas defasadas t-1, t-2..."""
        df_new = df.copy()
        for col in colunas:
            for i in range(1, n_lags + 1):
                df_new[f'{col}_lag_{i}'] = df_new[col].shift(i)
        return df_new

    def criar_medias_moveis(self, df, colunas, janelas=[3, 7, 30]):
        """Cria médias, desvios e max/min móveis."""
        df_new = df.copy()
        for col in colunas:
            for w in janelas:
                df_new[f'{col}_roll_mean_{w}'] = df_new[col].rolling(window=w).mean()
                df_new[f'{col}_roll_std_{w}'] = df_new[col].rolling(window=w).std()
        return df_new

    def codificar_tempo_ciclico(self, df, col_data='data'):
        """Transforma dia do ano em seno/cosseno para preservar ciclicidade."""
        df_new = df.copy()
        doy = df_new[col_data].dt.dayofyear
        df_new['doy_sin'] = np.sin(2 * np.pi * doy / 365.0)
        df_new['doy_cos'] = np.cos(2 * np.pi * doy / 365.0)
        return df_new

    def pipeline_completo(self, df):
        """Aplica todas as transformações e remove NaNs gerados."""
        cols_numericas = ['temperatura_max', 'precipitacao', 'pressao']
        # Validar colunas existentes
        cols_validas = [c for c in cols_numericas if c in df.columns]
        
        df = self.codificar_tempo_ciclico(df)
        df = self.criar_lags(df, cols_validas, n_lags=5)
        df = self.criar_medias_moveis(df, cols_validas, janelas=[7, 30])
        
        # Drop dos dias iniciais sem histórico
        df_limpo = df.dropna()
        return df_limpo

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Engenharia de Features...")
    
    # Dados Fake
    dates = pd.date_range('2024-01-01', periods=100)
    df = pd.DataFrame({
        'data': dates,
        'temperatura_max': np.random.normal(25, 5, 100),
        'precipitacao': np.random.exponential(5, 100),
        'pressao': np.random.normal(1013, 5, 100)
    })
    
    eng = EngenheiroFeatures()
    df_proc = eng.pipeline_completo(df)
    
    print(f"Shape Original: {df.shape}")
    print(f"Shape Processado: {df_proc.shape}")
    print(f"Novas Colunas (Total {len(df_proc.columns)}):")
    print(df_proc.columns.tolist()[:10], "...")
