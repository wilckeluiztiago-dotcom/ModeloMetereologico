import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA: ÍNDICE ULTRAVIOLETA (IUV)
====================================================

Calcula o IUV baseado na elevação solar e cobertura de nuvens.
Estimativa do UV Eritematoso (que queima a pele).

IUV = 0 a 2 (Baixo), ..., 11+ (Extremo).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class IndiceUV:
    def __init__(self):
        pass
        
    def calcular_iuv_ceu_claro(self, elevacao_solar_graus, ozonio_dobson=300):
        """
        Modelo simples baseado no ângulo zenital.
        """
        if elevacao_solar_graus <= 0: return 0.0
        
        zenite = 90 - elevacao_solar_graus
        cos_z = np.cos(np.radians(zenite))
        
        # Modelo aproximado
        # IUV0 ~ 12.5 * (cos_z)^2.42
        iuv_clear = 12.5 * (cos_z ** 2.42)
        
        # Correção Ozonio (RAF ~ 1.2) - Cada 1% menos ozonio, UV aumenta 1.2%
        fator_ozonio = (300 / ozonio_dobson) ** 1.2
        
        return iuv_clear * fator_ozonio

    def corrigir_nuvens(self, iuv_claro, cobertura_nuvens_octas):
        """
        Atenuação por nuvens.
        0 octas (claro) -> 100%
        8 octas (coberto) -> ~30%
        """
        fator = 1.0 - 0.09 * cobertura_nuvens_octas
        return iuv_claro * max(0.2, fator)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Índice UV...")
    calc = IndiceUV()
    
    # Meio dia verão (Sol a 80 graus)
    iuv = calc.calcular_iuv_ceu_claro(80, 280) # Buraco de ozonio leve
    print(f"IUV Céu Claro (Sol 80°): {iuv:.1f}")
    
    # Nublado
    iuv_nub = calc.corrigir_nuvens(iuv, 7) # 7/8 nuvens
    print(f"IUV Nublado: {iuv_nub:.1f}")
