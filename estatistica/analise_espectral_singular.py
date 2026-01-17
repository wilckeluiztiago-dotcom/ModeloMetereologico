import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
MÓDULO DE ANÁLISE ESPECTRAL SINGULAR (SSA)
==========================================

Técnica avançada não-paramétrica para decomposição de séries temporais.
SSA combina elementos de análise de séries temporais clássicas, estatística multivariada
(PCA), sistemas dinâmicos e processamento de sinais.

Excelente para extrair tendências e ciclos oscilatórios em dados ruidosos.

Algoritmo:
1. Embedding (Matriz de Trajetória - Hankel)
2. SVD (Decomposição de Valores Singulares)
3. Agrupamento (Grouping)
4. Reconstrução Diagonal (Diagonal Averaging)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class SingularSpectrumAnalysis:
    def __init__(self, window_size=12):
        self.L = window_size # Tamanho da janela (Embedding Dimension)
        self.ts = None
        self.N = 0
        self.K = 0
        self.X = None # Matriz de Trajetória
        self.U = None
        self.Sigma = None
        self.Vt = None
        
    def fit(self, time_series):
        """Executa a decomposição SVD."""
        self.ts = np.array(time_series)
        self.N = len(self.ts)
        self.K = self.N - self.L + 1
        
        if self.L > self.N // 2:
            print("AVISO: L deve ser <= N/2 para melhor separabilidade.")
            
        # 1. Embedding (Matriz Hankel)
        self.X = np.zeros((self.L, self.K))
        for i in range(self.K):
            self.X[:, i] = self.ts[i : i + self.L]
            
        # 2. SVD
        # X = U * Sigma * V.T
        self.U, self.Sigma, self.Vt = np.linalg.svd(self.X, full_matrices=False)
        
        # Variância explicada pelos componentes (Eigenvalues)
        self.eigenvalues = self.Sigma**2
        self.explained_variance = self.eigenvalues / np.sum(self.eigenvalues)
        
    def reconstruct(self, component_indices):
        """
        Reconstroi a série temporal usando apenas os componentes selecionados.
        (Passos 3 e 4)
        """
        # Reconstruir Matriz Elementar Xr
        Xr = np.zeros_like(self.X)
        
        for i in component_indices:
            # Componente i: sigma_i * u_i * v_i.T
            # Nota: numpy svd retorna Vt, então linha i de Vt
            Xi = self.Sigma[i] * np.outer(self.U[:, i], self.Vt[i, :])
            Xr += Xi
            
        # 4. Diagonal Averaging (Hankelization)
        rec_series = np.zeros(self.N)
        count = np.zeros(self.N)
        
        for i in range(self.L): # Linhas
            for j in range(self.K): # Colunas
                t = i + j
                rec_series[t] += Xr[i, j]
                count[t] += 1
                
        return rec_series / count

    def plot_w_correlation(self):
        """
        Plota matriz de W-correlação para verificar separabilidade dos componentes.
        (Avançado - Implementação simplificada aqui apenas da variância)
        """
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(self.explained_variance)), self.explained_variance * 100)
        plt.title('Espectro de Valores Singulares (Variância Explicada %)')
        plt.ylabel('% Variância')
        plt.xlabel('Índice do Componente')
        plt.grid(True, alpha=0.3)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Análise Espectral Singular (SSA)...")
    
    # Gerar série sintética
    t = np.linspace(0, 50, 200)
    
    # Componentes: Tendência + Ciclo 1 + Ciclo 2 + Ruído
    trend = 0.5 * t
    cycle1 = 5 * np.sin(2 * np.pi * t / 10)
    cycle2 = 2 * np.sin(2 * np.pi * t / 4)
    noise = np.random.normal(0, 1.5, len(t))
    
    serie = trend + cycle1 + cycle2 + noise
    
    # Ajuste SSA
    # L deve cobrir o maior ciclo de interesse. Ciclo 1 tem T=10.
    ssa = SingularSpectrumAnalysis(window_size=20) 
    ssa.fit(serie)
    
    print("SVD Calculado.")
    print(f"Top 5 componentes explicam: {ssa.explained_variance[:5]*100}")
    
    # Reconstrução
    # Geralmente comp 0 é tendência
    rec_trend = ssa.reconstruct([0])
    
    # Comps 1 e 2 devem ser o oscilador principal (pares em senoidais)
    rec_cycle = ssa.reconstruct([1, 2])
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, serie, 'gray', alpha=0.5, label='Original (Ruidosa)')
    plt.plot(t, rec_trend, 'r', linewidth=2, label='Reconstrução Tendência (Comp 0)')
    plt.plot(t, rec_trend + rec_cycle, 'g--', linewidth=2, label='Tendência + Ciclo Princ (Comp 0-2)')
    plt.legend()
    plt.title('Reconstrução de Sinal via SSA')
    print("Teste SSA concluído com sucesso.")
