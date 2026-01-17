import numpy as np

def calcular_balanco_energia_superficie(radiacao_solar_incidente, albedo, temperatura_superficie):
    """
    Calcula o balanço de energia simplificado na superfície.
    Rn = S_down * (1 - albedo) + L_down - L_up
    
    Onde:
    L_up = sigma * T^4 (Stefan-Boltzmann)
    """
    sigma = 5.67e-8
    
    # Estimativa de Radiação de Onda Longa incidente (L_down)
    # Empírico: depende da temperatura do ar e cobertura de nuvens
    # Simplificação: L_down ~ 0.8 * L_up (efeito estufa)
    
    l_up = sigma * (temperatura_superficie + 273.15)**4
    l_down = 0.8 * l_up 
    
    radiacao_liquida = radiacao_solar_incidente * (1 - albedo) + l_down - l_up
    
    return radiacao_liquida
