import numpy as np

def calcular_temperatura_potencial(temperatura_kelvin, pressao_hpa):
    """
    Calcula a Temperatura Potencial (Theta).
    Theta = T * (P0 / P)^(R/Cp)
    """
    P0 = 1000.0
    R_cp = 0.286 # R/Cp para ar seco
    
    theta = temperatura_kelvin * (P0 / pressao_hpa) ** R_cp
    return theta

def calcular_taxa_lapso_adiabatica(pressao, temperatura):
    """
    Estima a taxa de variação da temperatura com a altitude.
    """
    # Simplificação teórica
    g = 9.81
    cp = 1004.0
    gamma_d = g / cp # Taxa adiabática seca ~ 9.8 K/km
    return gamma_d
