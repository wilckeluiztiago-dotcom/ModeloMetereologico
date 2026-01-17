import numpy as np
import pandas as pd

"""
MÓDULO DE MÉTRICAS DE ERRO E VALIDAÇÃO
=======================================

Fornece uma bateria rigorosa de métricas para avaliar o desempenho
do modelo meteorológico comparando dados Simulado vs Observado.

Inclui métricas padrão (RMSE, MAE) e hidrológicas/climáticas (Nash-Sutcliffe, Willmott).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AvaliadorModelo:
    def __init__(self, observado, simulado):
        self.obs = np.array(observado)
        self.sim = np.array(simulado)
        self._validar()
        
    def _validar(self):
        if self.obs.shape != self.sim.shape:
            raise ValueError("Vetores de observação e simulação devem ter mesmo tamanho.")
        # Remover NaNs conjuntos
        mask = (~np.isnan(self.obs)) & (~np.isnan(self.sim))
        self.obs = self.obs[mask]
        self.sim = self.sim[mask]
        
    def rmse(self):
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((self.sim - self.obs)**2))
        
    def mae(self):
        """Mean Absolute Error."""
        return np.mean(np.abs(self.sim - self.obs))
        
    def bias(self):
        """Viés Médio (Sim - Obs)."""
        return np.mean(self.sim - self.obs)
        
    def mape(self):
        """Mean Absolute Percentage Error (%). Cuidado com zeros."""
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.abs((self.obs - self.sim) / self.obs)
            val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        return np.mean(val) * 100.0

    def nash_sutcliffe(self):
        """
        Coeficiente de Eficiência de Nash-Sutcliffe (NSE).
        NSE = 1 - (sum(obs - sim)^2 / sum(obs - mean_obs)^2)
        NSE = 1: Perfeito.
        NSE = 0: Modelo tão bom quanto a média observada.
        NSE < 0: Modelo pior que a média.
        """
        numerador = np.sum((self.obs - self.sim)**2)
        denominador = np.sum((self.obs - np.mean(self.obs))**2)
        
        if denominador == 0: return -np.inf
        return 1.0 - (numerador / denominador)

    def indice_willmott_d(self):
        """
        Índice de Concordância de Willmott (d).
        Varia de 0 (sem concordância) a 1 (concordância perfeita).
        """
        numerador = np.sum((self.sim - self.obs)**2)
        
        abs_sim = np.abs(self.sim - np.mean(self.obs))
        abs_obs = np.abs(self.obs - np.mean(self.obs))
        
        denominador = np.sum((abs_sim + abs_obs)**2)
        
        if denominador == 0: return 0.0
        return 1.0 - (numerador / denominador)

    def correlacao_pearson(self):
        """R de Pearson."""
        return np.corrcoef(self.obs, self.sim)[0, 1]

    def relatorio_completo(self):
        """Retorna string formatada com todas as métricas."""
        metricas = {
            "RMSE": self.rmse(),
            "MAE": self.mae(),
            "Bias": self.bias(),
            "MAPE (%)": self.mape(),
            "R (Pearson)": self.correlacao_pearson(),
            "R²": self.correlacao_pearson()**2,
            "Nash-Sutcliffe": self.nash_sutcliffe(),
            "Willmott d": self.indice_willmott_d()
        }
        
        txt = "=== Relatório de Performance do Modelo ===\n"
        for k, v in metricas.items():
            txt += f"{k:<20}: {v:.4f}\n"
        return txt

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Métricas de Erro...")
    
    # Caso 1: Modelo Perfeito
    obs1 = np.array([10, 12, 15, 14, 11])
    sim1 = np.array([10, 12, 15, 14, 11])
    av1 = AvaliadorModelo(obs1, sim1)
    print("\nCaso 1 - Perfeito (NSE deve ser 1.0):")
    print(av1.relatorio_completo())
    
    # Caso 2: Modelo com Viés (+2)
    obs2 = np.array([10, 12, 15, 14, 11])
    sim2 = np.array([12, 14, 17, 16, 13])
    av2 = AvaliadorModelo(obs2, sim2)
    print("\nCaso 2 - Com Viés Constante (RMSE=2.0, R=1.0):")
    print(av2.relatorio_completo())
    
    # Caso 3: Modelo Ruim Aleatório
    obs3 = np.random.normal(20, 5, 100)
    sim3 = np.random.normal(20, 5, 100) # Mesma média, mas não correlacionado
    av3 = AvaliadorModelo(obs3, sim3)
    print("\nCaso 3 - Aleatório (R~0, NSE<0 provável):")
    print(av3.relatorio_completo())
