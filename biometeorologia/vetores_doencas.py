import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE BIOMETEOROLOGIA: VETORES DE DOENÇAS (AEDES AEGYPTI)
=============================================================

Modelo epidemiológico-climático para estimar o risco de proliferação do mosquito
Aedes aegypti (Dengue, Zika, Chikungunya) baseado em Temperatura e Chuva.

Modela:
1. Taxa de desenvolvimento das larvas (Funcão da Temperatura).
2. Capacidade de suporte do ambiente (Função da Chuva acumulada).
3. Índice de Risco Vetorial.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloRiscoDengue:
    def __init__(self):
        pass
        
    def _taxa_desenvolvimento(self, temperatura):
        """
        Curva de performance térmica para o mosquito.
        Otimiza em ~28-30°C. Morre abaixo de 15°C e acima de 40°C.
        """
        if temperatura < 15 or temperatura > 40:
            return 0.0
        
        # Gaussiana centrada em 29°C
        taxa = np.exp(-((temperatura - 29)**2) / (2 * 5**2))
        return taxa

    def _fator_agua(self, chuva_acumulada_15dias):
        """
        Disponibilidade de criadouros.
        Precisa de chuva para encher potes, mas chuva excessiva pode lavar larvas.
        """
        # Saturação logística
        # 0mm -> 0
        # 50mm -> 0.8
        # >100mm -> 1.0 (mas com risco de washout se muito intenfo, ignorado aqui)
        return 1 - np.exp(-0.03 * chuva_acumulada_15dias)

    def calcular_indice_risco(self, t_media, chuva_15d):
        """
        Retorna Índice de Risco (0 a 1).
        Combinacao multiplicativa (ambas condições necessárias).
        """
        f_temp = self._taxa_desenvolvimento(t_media)
        f_agua = self._fator_agua(chuva_15d)
        
        risco = f_temp * f_agua * 100 # Escala 0-100
        return risco

    def classificar_risco(self, indice):
        if indice < 10: return "Baixo"
        if indice < 40: return "Médio"
        if indice < 70: return "Alto"
        return "Epidêmico"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Modelo de Risco de Dengue...")
    
    modelo = ModeloRiscoDengue()
    
    # Cenários
    cenarios = [
        ("Inverno Seco", 12, 10),
        ("Verão Seco", 30, 5),
        ("Verão Chuvoso", 28, 80), # Cenário Ideal
        ("Onda Calor Extremo", 42, 50) # Muito quente mata mosquito
    ]
    
    print(f"{'Cenário':<20} {'Temp':<5} {'Chuva':<5} {'Risco':<10} {'Classe'}")
    for nome, t, c in cenarios:
        r = modelo.calcular_indice_risco(t, c)
        classe = modelo.classificar_risco(r)
        print(f"{nome:<20} {t:<5} {c:<5} {r:<10.1f} {classe}")
        
    # Mapa de calor T vs Chuva
    ts = np.linspace(10, 40, 50)
    cs = np.linspace(0, 150, 50)
    TT, CC = np.meshgrid(ts, cs)
    RR = np.zeros_like(TT)
    
    for i in range(50):
        for j in range(50):
            RR[i,j] = modelo.calcular_indice_risco(TT[i,j], CC[i,j])
            
    plt.figure(figsize=(8, 6))
    plt.contourf(TT, CC, RR, levels=20, cmap='RdYlGn_r') # Verde=Baixo, Vermelho=Alto
    plt.colorbar(label='Índice de Risco (0-100)')
    plt.xlabel('Temperatura Média (°C)')
    plt.ylabel('Chuva Acumulada 15 dias (mm)')
    plt.title('Diagrama de Risco Climático para Dengue')
    plt.axvline(29, color='k', linestyle='--', alpha=0.3, label='Ótimo Térmico')
    print("Gráfico Epidemiológico gerado.")
