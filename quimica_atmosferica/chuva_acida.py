import numpy as np

"""
MÓDULO DE QUÍMICA ATMOSFÉRICA: CHUVA ÁCIDA (Estimativa de pH)
=============================================================

Estima o pH da precipitação baseada na concentração de SO2 e NO2 na atmosfera
que são absorvidos pelas gotas (Formação de H2SO4 e HNO3).

pH Água Pura (Eq com CO2) ~= 5.6
pH Chuva Ácida < 5.0 (Crítico < 4.5)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ChuvaAcida:
    def __init__(self):
        pass
        
    def estimar_ph(self, conc_so2_ppb, conc_no2_ppb, chuva_mm):
        """
        Modelo empírico simplificado.
        Concentrações altas de SO2/NO2 reduzem pH.
        Chuva intensa dilui (aumenta pH em direção ao neutro).
        """
        # pH base (água em equilíbrio com CO2 atm)
        ph = 5.6
        
        # Produção de íons H+ (Acidificação)
        # H+ proportional to SO2 + 0.5*NO2
        carga_acida = (conc_so2_ppb * 1.5 + conc_no2_ppb * 0.5)
        
        # Fator de Diluição (Quanto mais chuva, menos ácido concentrado)
        fator_diluicao = np.log1p(chuva_mm) + 1.0 # log(1+x) evita div zero
        
        # Redução do pH
        # Escala ajustada empiricamente para dar valores realistas (3.5 a 5.6)
        delta_ph = (carga_acida / fator_diluicao) * 0.05
        
        ph_final = ph - delta_ph
        return max(3.0, ph_final) # Limite inferior físico aproximado

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Modelo de Chuva Ácida...")
    
    modelo = ChuvaAcida()
    
    # Cenário: Cubatão anos 80 (Muito poluído)
    so2 = 50.0 # ppb
    no2 = 40.0 # ppb
    chuva = 5.0 # mm (garoa ácida)
    
    ph = modelo.estimar_ph(so2, no2, chuva)
    print(f"Cenário Poluído -> pH: {ph:.2f}")
    
    # Cenário: Campo limpo
    ph_limpo = modelo.estimar_ph(1.0, 2.0, 20.0)
    print(f"Cenário Limpo -> pH: {ph_limpo:.2f}")
