import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA E SEGURANÇA: RISCO DE INCÊNDIO FLORESTAL (FMA)
========================================================================

Implementa a Fórmula de Monte Alegre (FMA), padrão brasileiro para risco de fogo.
Baseia-se na Umidade Relativa às 13h e na Chuva acumulada.

FMA acumula diariamente se não chover.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class FormulaMonteAlegre:
    def __init__(self):
        self.fma_acum = 0.0
        
    def passo_diario(self, ur_13h, chuva_mm):
        """
        Calcula FMA do dia.
        UR em %.
        """
        # Calcular H (Índice diário base)
        h = 0.0
        if ur_13h < 25: h = 100 - 2.5 * ur_13h # Muito seco
        elif ur_13h < 45: h = 87 - 2 * ur_13h
        else: h = 47 - ur_13h
        if h < 0: h = 0 # Umidade alta não soma risco
        
        # Atualizar acumulado
        self.fma_acum += h
        
        # Redução pela chuva
        if chuva_mm >= 2.0:
            if chuva_mm < 5.0: self.fma_acum *= 0.7
            elif chuva_mm < 10.0: self.fma_acum *= 0.4
            elif chuva_mm < 20.0: self.fma_acum *= 0.2
            else: self.fma_acum = 0.0 # Zerou risco
            
        return self.fma_acum
        
    def classificar(self, fma):
        if fma <= 2.0: return "Nulo"
        if fma <= 4.0: return "Pequeno"
        if fma <= 8.0: return "Médio"
        if fma <= 15.0: return "Alto"
        return "Muito Alto"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando FMA (Risco de Fogo)...")
    fma = FormulaMonteAlegre()
    
    # 5 dias secos (UR 30%)
    for i in range(5):
        val = fma.passo_diario(30, 0)
        print(f"Dia {i+1}: FMA={val:.1f} ({fma.classificar(val)})")
