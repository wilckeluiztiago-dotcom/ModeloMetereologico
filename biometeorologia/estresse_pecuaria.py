import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA ANIMAL: ESTRESSE POR FRIO E CALOR EM PECUÁRIA
=======================================================================

Calcula o Índice de Temperatura e Umidade (ITU/THI) adaptado para bovinos.
ITU = 0.8*Ta + RH/100*(Ta - 14.4) + 46.4

Faixas (Holandesa):
< 72: Conforto
72-78: Alerta
79-88: Perigo
> 89: Emergência

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloPecuaria:
    def __init__(self):
        pass
        
    def calcular_itu(self, temp_c, hum_rel):
        """Índice de Temperatura e Umidade (Thom, 1959)."""
        itu = 0.8 * temp_c + (hum_rel / 100.0) * (temp_c - 14.4) + 46.4
        return itu
        
    def perda_leite_estimada(self, itu):
        """Estimativa empírica de perda de produção (kg/dia)"""
        if itu < 72: return 0.0
        # Perda cresce exponencial
        perda = 0.2 * (itu - 72) ** 1.5 
        return perda

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando ITU Pecuária...")
    mod = ModeloPecuaria()
    
    itu = mod.calcular_itu(32, 80) # Verão abafado
    perda = mod.perda_leite_estimada(itu)
    
    print(f"ITU: {itu:.1f}")
    print(f"Perda Leite: {perda:.2f} kg/vaca/dia")
