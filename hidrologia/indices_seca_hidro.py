import numpy as np
import scipy.stats as stats

"""
MÓDULO DE HIDROLOGIA: ÍNDICES DE SECA HIDROLÓGICA (SRI)
=======================================================

Cálculo do Standardized Runoff Index (SRI), análogo ao SPI meteorológico.
Quantifica o déficit de vazão em termos probabilísticos (desvios padrão).
SRI < -1.5 indica seca severa nos rios.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class IndicesSecaHidro:
    def __init__(self):
        pass
        
    def calcular_sri(self, serie_vazoes_mensais):
        """
        Calcula SRI ajustando uma distribuição Gamma ou Log-Normal à série
        e convertendo para Normal Padrão (Z-score).
        """
        # Filtrar zeros (Gamma nao aceita 0)
        vazoes_validas = np.array(serie_vazoes_mensais)
        vazoes_validas[vazoes_validas <= 0] = 0.01 # Pequeno epsilon
        
        # Ajuste Gamma (padrão)
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(vazoes_validas)
        
        # Calcular Probabilidade Acumulada (CDF)
        cdf = stats.gamma.cdf(vazoes_validas, fit_alpha, loc=fit_loc, scale=fit_beta)
        
        # Converter para Z-score (Inverse Normal CDF)
        sri = stats.norm.ppf(cdf)
        
        # Tratar infs possíveis nos extremos
        sri = np.nan_to_num(sri, nan=0.0)
        return sri

    def classificar_sri(self, valor_sri):
        if valor_sri >= 2.0: return "Extremamente Úmido"
        if valor_sri >= 1.5: return "Muito Úmido"
        if valor_sri >= 1.0: return "Moderadamente Úmido"
        if valor_sri > -1.0: return "Normal"
        if valor_sri > -1.5: return "Seca Moderada"
        if valor_sri > -2.0: return "Seca Severa"
        return "Seca Extrema"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Cálculo de SRI (Seca Hidrológica)...")
    
    ind = IndicesSecaHidro()
    
    # Gerar série sintética (Gamma)
    vazoes = np.random.gamma(shape=2, scale=50, size=120) # 10 anos
    
    # Inserir seca artificial
    vazoes[60:72] = vazoes[60:72] * 0.2
    
    sri = ind.calcular_sri(vazoes)
    
    print("\nAnálise de Seca (Trecho com falha):")
    for i in range(60, 65):
        print(f"Mês {i}: Vazão={vazoes[i]:.1f}, SRI={sri[i]:.2f} ({ind.classificar_sri(sri[i])})")
        
    print(f"\nMínimo SRI atingido: {np.min(sri):.2f}")
