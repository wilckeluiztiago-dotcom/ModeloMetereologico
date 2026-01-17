import numpy as np

"""
MÓDULO DE QUÍMICA ATMOSFÉRICA: SMOG FOTOQUÍMICO (ÍNDICE)
========================================================

Estima o potencial de formação de Smog (Nevoeiro fotoquímico marrom).
Combina Potencial Oxidante (Ox = O3 + NO2) e visibilidade reduzida por aerossóis.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class IndicadorSmog:
    def __init__(self):
        pass
        
    def calcular_potencial_oxidante(self, o3_ppb, no2_ppb):
        """Soma de oxidantes fotoquímicos."""
        ox = o3_ppb + no2_ppb
        return ox

    def classificar_visibilidade(self, pm25_ugm3, umidade_rel):
        """
        Estima visibilidade visual (Koschmieder equation).
        Bext = 3.912 / Vis (km)
        Bext ~ alpha * PM2.5 * f(RH) (fator de crescimento higroscópico)
        """
        # Fator de crescimento (sulfatos/nitratos incham com umidade)
        f_rh = 1.0 + 0.3 * (umidade_rel / 100.0)**2 
        if umidade_rel > 90: f_rh *= 2 # Nevoeiro mesmo
        
        # Extinção (1/km)
        # 0.003 é eficiência mássica específica dry aproximada
        b_ext = 0.01 + 0.003 * pm25_ugm3 * f_rh
        
        visibilidade_km = 3.912 / b_ext
        return min(300, visibilidade_km)

    def diagnostico_smog(self, o3, no2, pm25, rh):
        ox = self.calcular_potencial_oxidante(o3, no2)
        vis = self.classificar_visibilidade(pm25, rh)
        
        status = "Limpo"
        if ox > 80: status = "Leve Smog"
        if ox > 120 and vis < 10: status = "Smog Moderado"
        if ox > 150 and vis < 5: status = "Smog Severo (Perigoso)"
        
        return ox, vis, status

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Indicador de Smog...")
    
    ind = IndicadorSmog()
    
    # Dia feio em SP ou Santiago
    o3 = 100
    no2 = 60
    pm = 80
    ur = 60
    
    ox, vis, stat = ind.diagnostico_smog(o3, no2, pm, ur)
    
    print(f"Potencial Oxidante: {ox} ppb")
    print(f"Visibilidade Estimada: {vis:.1f} km")
    print(f"Diagnóstico: {stat}")
