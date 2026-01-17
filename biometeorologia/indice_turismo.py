import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA: ÍNDICE DE TURISMO CLIMÁTICO (TCI)
============================================================

Adaptado de Mieczkowski (1985). Avalia a adequação climática para turismo ao ar livre.
Pesos:
- Conforto Térmico (Dia) [40%]
- Conforto Térmico (Dia+Noite) [10%]
- Precipitação [20%]
- Insolação [20%]
- Vento [10%]

TCI varia de -30 a 100 (Ideal).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class IndiceTurismo:
    def __init__(self):
        pass
        
    def _pontuar_chuva(self, mm_mes):
        """Menos chuva melhor."""
        if mm_mes < 15: return 5.0
        if mm_mes < 30: return 4.5
        if mm_mes < 45: return 4.0
        if mm_mes < 60: return 3.0
        if mm_mes < 90: return 2.0
        if mm_mes < 120: return 1.0
        return 0.0 # Muita chuva = ruim

    def _pontuar_conforto(self, temp_max):
        """Temperatura ideal 20-27C."""
        if 20 <= temp_max <= 27: return 5.0
        if 18 <= temp_max < 20 or 27 < temp_max <= 29: return 4.5
        if temp_max > 35 or temp_max < 5: return 1.0
        return 3.0

    def calcular_tci_mensal(self, t_max_media, t_media, chuva_mm_total, sol_horas_dia, vento_ms):
        """
        Cálculo simplificado do TCI.
        Retorna Valor TCI e Categoria.
        """
        # Sub-indices (0 a 5)
        cid = self._pontuar_conforto(t_max_media) # Conforto diurno (40%) - peso 2 (escala 10) -> 8 pts max
        cia = self._pontuar_conforto(t_media) # Conforto diario (10%)
        
        # Chuva (20%)
        p_chuva = self._pontuar_chuva(chuva_mm_total)
        
        # Sol (20%) - Mais sol melhor, até certo ponto
        p_sol = min(5.0, sol_horas_dia / 2.0)
        
        # Vento (10%) - Vento leve é bom, forte é ruim
        p_vento = 5.0
        if vento_ms > 6: p_vento = 2.0
        if vento_ms > 10: p_vento = 0.0
        
        # Fórmula TCI = 4*CID + 1*CIA + 2*P_Chuva + 2*P_Sol + 1*P_Vento (Soma max = 20 + 5 + 10 + 10 + 5 = 50 * 2 = 100)
        tci = 2 * (4*cid + 1*cia + 2*p_chuva + 2*p_sol + 1*p_vento)
        
        return tci

    def classificar_tci(self, tci):
        if tci >= 80: return "Excelente"
        if tci >= 70: return "Muito Bom"
        if tci >= 60: return "Bom"
        if tci >= 50: return "Aceitável"
        if tci >= 40: return "Marginal"
        return "Desfavorável"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Índice de Turismo (TCI)...")
    
    tci_calc = IndiceTurismo()
    
    # Florianópolis em Março (Bom)
    score = tci_calc.calcular_tci_mensal(
        t_max_media=26,
        t_media=23,
        chuva_mm_total=120, # Chove um pouco
        sol_horas_dia=7,
        vento_ms=3
    )
    
    cat = tci_calc.classificar_tci(score)
    
    print(f"Florianópolis (Mar): TCI={score:.1f} ({cat})")
    
    # Serra Gaúcha em Julho (Frio/Chuva)
    score_inv = tci_calc.calcular_tci_mensal(
        t_max_media=12,
        t_media=8,
        chuva_mm_total=180,
        sol_horas_dia=4,
        vento_ms=5
    )
    cat_inv = tci_calc.classificar_tci(score_inv)
    
    print(f"Serra Gaúcha (Jul): TCI={score_inv:.1f} ({cat_inv})")
