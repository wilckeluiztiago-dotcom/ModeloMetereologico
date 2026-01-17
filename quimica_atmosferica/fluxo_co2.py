import numpy as np

"""
MÓDULO DE QUÍMICA ATMOSFÉRICA: FLUXO DE CO2 BIOSFÉRICO
======================================================

Estima a Troca Líquida do Ecossistema (NEE - Net Ecosystem Exchange) de CO2.
NEE = Respiração (R) - Fotossíntese (GPP).

Modelos:
- GPP: Dependência de Luz (Michaelis-Menten) e Temperatura.
- Respiração: Q10 (Dependência exponencial da temperatura do solo).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloFluxoCO2:
    def __init__(self, tipo_vegetacao='floresta'):
        # Parâmetros para Floresta Subtropical
        if tipo_vegetacao == 'floresta':
            self.gpp_max = 30.0 # micromol/m2/s
            self.alpha = 0.05   # Eficiência quântica
            self.r_base = 2.0   # Respiração a 20C
            self.q10 = 2.0      # Fator de aumento
        else: # Soja/Campo
            self.gpp_max = 45.0
            self.alpha = 0.08
            self.r_base = 1.5
            self.q10 = 1.8
            
    def calcular_gpp(self, par, temperatura):
        """
        Gross Primary Production (Fotossíntese).
        PAR: Photosynthetically Active Radiation (W/m2).
        """
        # Limitação por Luz (Hipérbole Retangular)
        gpp_luz = (self.alpha * par * self.gpp_max) / (self.alpha * par + self.gpp_max)
        
        # Limitação por Temperatura (Parábola simples)
        # Ótimo em 25C, zero < 0 e > 40
        temp_factor = max(0, 1 - ((temperatura - 25)/15)**2)
        
        return gpp_luz * temp_factor

    def calcular_respiracao_solo(self, temp_solo):
        """
        Respiração ecossistêmica (Solo + Plantas).
        Modelo Q10.
        """
        re = self.r_base * (self.q10 ** ((temp_solo - 20) / 10.0))
        return re

    def calcular_nee(self, par, temp_ar, temp_solo):
        """
        Net Ecosystem Exchange.
        Sinal: Negativo = Sumidouro (Absorção), Positivo = Fonte (Emissão).
        """
        gpp = self.calcular_gpp(par, temp_ar)
        re = self.calcular_respiracao_solo(temp_solo)
        
        nee = re - gpp # Convenção atmosférica (fluxo para atmosfera)
        return nee

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Fluxo de CO2...")
    
    modelo = ModeloFluxoCO2()
    
    # Dia ensolarado
    par = 500 # W/m2 visível
    t_ar = 28
    t_solo = 22
    
    nee_dia = modelo.calcular_nee(par, t_ar, t_solo)
    
    # Noite
    nee_noite = modelo.calcular_nee(0, 15, 18)
    
    print(f"NEE Dia: {nee_dia:.2f} µmol/m²/s (Absorção)" if nee_dia < 0 else f"NEE Dia: {nee_dia:.2f} (Emissão)")
    print(f"NEE Noite: {nee_noite:.2f} µmol/m²/s (Emissão)")
