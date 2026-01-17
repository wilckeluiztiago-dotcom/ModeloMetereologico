import pandas as pd
import numpy as np
from estatistica.teste_mann_kendall import teste_mann_kendall

def analisar_impacto_el_nino_sul(df_chuva, indice_nino):
    """
    Correlaciona índice El Niño com chuvas no Sul.
    No Sul, El Niño geralmente aumenta a chuva.
    """
    # Assumindo mesmo comprimento para simplificação
    min_len = min(len(df_chuva), len(indice_nino))
    corr = np.corrcoef(df_chuva[:min_len], indice_nino[:min_len])[0, 1]
    
    return corr
