import numpy as np

"""
MÓDULO DE HIDROLOGIA E CRIOSFERA: DERRETIMENTO DE NEVE (DEGREE-DAY)
===================================================================

Estimativa de derretimento de neve acumulada nas serras (RS/SC) baseada na 
temperatura do ar acima de um limiar (Degree-Day Factor).

Melt = DDF * (Temp - T_base)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloNeve:
    def __init__(self):
        self.t_base = 0.0 # Temperatura crítica de derretimento (°C)
        self.ddf = 3.0    # Degree-Day Factor (mm/°C/dia) - Típico para neve sazonal
        
    def passo_tempo(self, estoque_neve_mm, precipitacao_mm, temperatura_media):
        """
        Atualiza o estoque de neve e calcula água de degelo.
        precipitacao_mm: Total (chuva + neve).
        temperatura_media: °C.
        """
        novo_estoque = estoque_neve_mm
        precip_solida = 0.0
        precip_liquida = 0.0
        derretimento = 0.0
        
        # 1. Partição da Precipitação (Chuva vs Neve)
        if temperatura_media <= 1.0: # Limiar aproximado neve/chuva
            precip_solida = precipitacao_mm
            novo_estoque += precip_solida
        else:
            precip_liquida = precipitacao_mm
            
        # 2. Derretimento (Ablação)
        if temperatura_media > self.t_base:
            potencial_derretimento = self.ddf * (temperatura_media - self.t_base)
            
            # Não pode derreter mais do que tem
            derretimento = min(novo_estoque, potencial_derretimento)
            novo_estoque -= derretimento
            
        # Água total disponível no solo (Chuva + Degelo)
        agua_para_hidrologia = precip_liquida + derretimento
        
        return novo_estoque, derretimento, agua_para_hidrologia

    def simular_evento_frio(self, temps, precips):
        """Simula uma onda de frio com neve posterior aquecimento."""
        estoque = 0.0
        hist_estoque = []
        hist_degelo = []
        
        for t, p in zip(temps, precips):
            estoque, degelo, _ = self.passo_tempo(estoque, p, t)
            hist_estoque.append(estoque)
            hist_degelo.append(degelo)
            
        return hist_estoque, hist_degelo

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Modelo de Neve (Degree-Day)...")
    
    modelo = ModeloNeve()
    
    # Cenário: 3 dias frio (-2C, 0C, 1C) com precip, depois calor (5C, 10C)
    temps = [-2, -1, 0, 5, 10, 12]
    precips = [10, 20, 5, 0, 0, 0]
    
    est, deg = modelo.simular_evento_frio(temps, precips)
    
    for i, (t, p, e, d) in enumerate(zip(temps, precips, est, deg)):
        print(f"Dia {i+1}: T={t}C, P={p}mm -> Neve={e:.1f}mm, Degelo={d:.1f}mm")
