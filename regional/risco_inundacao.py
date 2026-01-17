import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

"""
MÓDULO DE RISCO DE INUNDAÇÃO (HIDROLOGIA ESTATÍSTICA)
=====================================================

Aplica a Teoria de Valores Extremos (Distribuição de Gumbel) para calcular
períodos de retorno de chuvas intensas e estimar vazões de pico
usando o Método Racional Modificado.

Foco: Prevenção de desastres no Vale do Taquari (RS) e Vale do Itajaí (SC).

Equações:
1. IDF (Intensidade-Duração-Frequência): i = K * T^a / (t + b)^c
2. Método Racional: Q = C * i * A / 3.6

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AnalisadorRiscoInundacao:
    def __init__(self):
        # Parâmetros IDF típicos para Porto Alegre (Exemplo)
        # i em mm/h, T em anos, t em minutos
        self.idf_params = {'K': 900, 'a': 0.18, 'b': 12, 'c': 0.75}
        
    def calcular_intensidade_chuva(self, tempo_retorno_anos, duracao_minutos):
        """Calcula intensidade da chuva (mm/h) pela curva IDF."""
        K, a, b, c = self.idf_params.values()
        i = (K * (tempo_retorno_anos ** a)) / ((duracao_minutos + b) ** c)
        return i

    def estimar_vazao_pico(self, tempo_retorno, area_bacia_km2, tempo_concentracao_min, coef_runoff=0.7):
        """
        Calcula vazão de pico (Q) em m³/s pelo Método Racional.
        """
        # Intensidade média na duração igual ao tempo de concentração
        intensidade = self.calcular_intensidade_chuva(tempo_retorno, tempo_concentracao_min)
        
        # Q = C * I * A / 3.6
        # C: adimensional (0-1)
        # I: mm/h
        # A: km²
        # 3.6: Fator de conversão
        q_pico = (coef_runoff * intensidade * area_bacia_km2) / 3.6
        return q_pico, intensidade

    def ajustar_gumbel_maximos(self, serie_vazoes_maximas):
        """
        Ajusta distribuição de Gumbel aos dados de vazão máxima anual.
        Retorna parâmetros (loc, scale) e função para calcular retorno.
        Prob(X <= x) = exp(-exp(-(x - loc)/scale))
        """
        loc, scale = gumbel_r.fit(serie_vazoes_maximas)
        return loc, scale

    def calcular_nivel_retorno_gumbel(self, loc, scale, return_period):
        """Calcula o valor associado a um período de retorno T."""
        prob = 1 - 1/return_period
        valor = gumbel_r.ppf(prob, loc, scale)
        return valor

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Análise de Risco de Inundação...")
    
    analisador = AnalisadorRiscoInundacao()
    
    # 1. Simulação: Bacia do Rio Taquari (Simplificada)
    area = 26000 # km² (Enorme, método racional não ideal, mas serve para demo)
    tc = 1440 # 24 horas de concentração
    
    print(f"\nBacia: Area={area} km², Tc={tc/60:.1f} horas")
    
    # Calcular Q para diferentes TR
    periodos = [2, 5, 10, 50, 100, 1000]
    
    print("-" * 50)
    print(f"{'TR (Anos)':<10} {'Chuva (mm/h)':<15} {'Vazão Pico (m³/s)':<20}")
    print("-" * 50)
    
    vazoes_simuladas = []
    
    for tr in periodos:
        q, i = analisador.estimar_vazao_pico(tr, area, tc, coef_runoff=0.6)
        print(f"{tr:<10} {i:<15.2f} {q:<20.0f}")
        vazoes_simuladas.append(q)
        
    print("-" * 50)
    
    # 2. Ajuste Estatístico em Dados Sintéticos de Vazão (1990-2024)
    # Supondo histórico de 34 anos
    # Gerar vazoes Gumbel aleatórias
    np.random.seed(42)
    historico_vazoes = np.random.gumbel(loc=5000, scale=1500, size=34)
    
    loc_est, scale_est = analisador.ajustar_gumbel_maximos(historico_vazoes)
    
    print(f"\nAjuste Gumbel (Dados Históricos Sintéticos):")
    print(f"Parâmetros: u={loc_est:.2f}, beta={scale_est:.2f}")
    
    q_100_est = analisador.calcular_nivel_retorno_gumbel(loc_est, scale_est, 100)
    print(f"Vazão estimada para 100 anos (baseado no histórico): {q_100_est:.0f} m³/s")
    
    # Plot PDF Gumbel
    x = np.linspace(0, 15000, 200)
    pdf = gumbel_r.pdf(x, loc_est, scale_est)
    
    plt.figure(figsize=(10, 6))
    plt.hist(historico_vazoes, density=True, alpha=0.5, label='Histórico (34 anos)')
    plt.plot(x, pdf, 'r-', linewidth=2, label='PDF Gumbel Ajustada')
    plt.title('Distribuição de Vazões Máximas Anuais')
    plt.xlabel('Vazão (m³/s)')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    # plt.show()
    print("Gráfico de risco gerado.")
