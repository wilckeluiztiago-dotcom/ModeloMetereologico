import numpy as np

"""
MÓDULO DE HIDROLOGIA: EVAPOTRANSPIRACAO DE PENMAN-MONTEITH (FAO-56)
===================================================================

Implementa o método padrão ouro da FAO para cálculo da Evapotranspiração de Referência (ETo).
Combina o balanço de energia (radiação) com o balanço aerodinâmico (vento/umidade).

Equação complexa que requer:
- Temperatura média, max, min
- Umidade Relativa
- Velocidade do Vento a 2m
- Radiação Solar Líquida (Rn)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloPenmanMonteith:
    def __init__(self, latitude_graus=-30.0, altitude_m=100.0):
        self.lat = np.radians(latitude_graus)
        self.z = altitude_m
        
        # Pressão atmosférica local (kPa)
        self.P = 101.3 * ((293 - 0.0065 * self.z) / 293) ** 5.26
        
        # Constante psicrométrica (gamma) ~ 0.063 kPa/°C
        self.gamma = 0.665e-3 * self.P
        
    def _pressao_vapor_saturacao(self, t):
        """Tetens: e0(T) em kPa."""
        return 0.6108 * np.exp((17.27 * t) / (t + 23.73))

    def _declividade_curva_pressao(self, t):
        """Slope (Delta) da curva de saturação em kPa/°C."""
        e0 = self._pressao_vapor_saturacao(t)
        return (4098 * e0) / ((t + 23.73) ** 2)

    def calcular_eto_diario(self, t_min, t_max, ur_media, vento_2m, radiacao_solar_mj_m2):
        """
        Calcula ETo (mm/dia) para grama de referência.
        """
        t_media = (t_max + t_min) / 2.0
        
        # 1. Termo de Radiação
        # Converter Global (Rs) para Liquida (Rn) - Simplificado
        # Rn = 0.77 * Rs (Albedo grama 0.23) - Onda Longa Liquida (~ fixo)
        rn = 0.77 * radiacao_solar_mj_m2 - 2.0 # Aprox grosseira para Onda Longa Out
        
        # Transformar Rn de MJ/m2/dia para equivalencia de mm/dia? 
        # A eq Penman-Monteith usa Rn direta no numerador com coef 0.408
        
        # 2. Termo Aerodinâmico
        delta = self._declividade_curva_pressao(t_media)
        
        es_max = self._pressao_vapor_saturacao(t_max)
        es_min = self._pressao_vapor_saturacao(t_min)
        es = (es_max + es_min) / 2.0
        
        ea = es * (ur_media / 100.0) # Pressão real
        
        # Déficit de pressão de vapor
        dpv = es - ea
        
        # Numerador
        num_rad = 0.408 * delta * max(0, rn)
        num_aero = self.gamma * (900 / (t_media + 273)) * vento_2m * dpv
        
        # Denominador
        den = delta + self.gamma * (1 + 0.34 * vento_2m)
        
        eto = (num_rad + num_aero) / den
        
        return max(0, eto) # Não pode ser negativo

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Evapotranspiração Penman-Monteith...")
    
    pm = ModeloPenmanMonteith(latitude_graus=-30, altitude_m=10)
    
    # Dia de Verão Quente e Seco
    eto_verao = pm.calcular_eto_diario(
        t_min=22, t_max=35, ur_media=40, vento_2m=3.5, radiacao_solar_mj_m2=28
    )
    
    # Dia de Inverno Frio e Úmido
    eto_inverno = pm.calcular_eto_diario(
        t_min=5, t_max=15, ur_media=85, vento_2m=1.0, radiacao_solar_mj_m2=10
    )
    
    print(f"ETo Verão (Esperado > 6mm): {eto_verao:.2f} mm/dia")
    print(f"ETo Inverno (Esperado < 2mm): {eto_inverno:.2f} mm/dia")
