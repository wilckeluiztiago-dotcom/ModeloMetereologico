import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: RECUPERAÇÃO DE TEMPERATURA DE SUPERFÍCIE (LST)
==============================================================================

Algoritmo Split-Window para estimar LST a partir de duas bandas térmicas (T11 e T12).
Baseado no algoritmo de Wan & Dozier (MODIS).

LST = T11 + C1*(T11-T12) + C2*(T11-T12)^2 + C0 + (1-eps)/eps...

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class RecuperacaoLST:
    def __init__(self):
        pass
        
    def calcular_lst_split_window(self, t11_kelvin, t12_kelvin, emissividade_media=0.98):
        """
        Versão simplificada linear.
        """
        # Diferença de temperatura
        delta_t = t11_kelvin - t12_kelvin
        
        # Coeficientes empíricos (para água/vegetação)
        # T_s = T11 + 1.8 * (T11 - T12) + 48*(1-eps) - 75*delta_eps
        
        # Correção Emissividade
        corr_eps = 50 * (1 - emissividade_media)
        
        lst = t11_kelvin + 2.0 * delta_t + corr_eps
        return lst

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Retrieval LST...")
    rec = RecuperacaoLST()
    
    # Exemplo: Vegetação
    t11 = 300.0 # Banda 11 microns
    t12 = 298.0 # Banda 12 microns (absorve mais vapor d'agua)
    
    lst = rec.calcular_lst_split_window(t11, t12, 0.98)
    print(f"Brilho T11: {t11}K, T12: {t12}K")
    print(f"LST Estimada: {lst:.2f}K ({lst-273.15:.2f}°C)")
