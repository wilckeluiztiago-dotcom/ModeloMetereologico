import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
MÓDULO DE INTERPOLAÇÃO ESPACIAL (KRIGAGEM)
==========================================

Implementa Krigagem Ordinária (Ordinary Kriging) do zero para espacializar dados
de estações pontuais para uma grade regular no mapa do Sul do Brasil.
A Krigagem é o 'Blue Best Linear Unbiased Estimator' (BLUE).

Funde álgebra linear pesada para resolver os pesos lambda baseados no variograma.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class KrigagemSimples:
    def __init__(self, modelo_variograma='esferico', alcance=5.0, patamar=10.0, pepita=1.0):
        """
        Args:
            modelo_variograma (str): 'esferico', 'exponencial', 'gaussiano'
            alcance (float): Range (distância onde cessa correlação)
            patamar (float): Sill (variância total)
            pepita (float): Nugget (erro na origem)
        """
        self.modelo = modelo_variograma
        self.a = alcance
        self.c0 = patamar
        self.c_n = pepita
        
        self.X_treino = None
        self.y_treino = None
        self.K_inv = None # Matriz de covariância invertida
        
    def _variograma_func(self, h):
        """Calcula gamma(h) - Semivariância."""
        h = np.abs(h)
        val = np.zeros_like(h)
        
        # Efeito pepita
        mask_nugget = h > 0
        val[~mask_nugget] = 0
        
        if self.modelo == 'esferico':
            # 1.5*(h/a) - 0.5*(h/a)^3 para h < a
            mask = (h <= self.a) & (h > 0)
            val[mask] = self.c_n + (self.c0 - self.c_n) * (1.5*(h[mask]/self.a) - 0.5*(h[mask]/self.a)**3)
            val[h > self.a] = self.c0 # Patamar
            
        elif self.modelo == 'exponencial':
            # c * (1 - exp(-3h/a))
            # Ajuste clássico onde alcance efetivo visual é a
            val[mask_nugget] = self.c_n + (self.c0 - self.c_n) * (1 - np.exp(-3 * h[mask_nugget] / self.a))
            
        elif self.modelo == 'gaussiano':
            val[mask_nugget] = self.c_n + (self.c0 - self.c_n) * (1 - np.exp(-3 * (h[mask_nugget]/self.a)**2))
            
        return val

    def _covariancia_func(self, h):
        """C(h) = Sill - Gamma(h)"""
        return self.c0 - self._variograma_func(h)

    def ajustar(self, coordenadas, valores):
        """
        Prepara as matrizes de krigagem.
        Coordenadas: array (N, 2) [lat, lon]
        Valores: array (N,)
        """
        self.X_treino = np.array(coordenadas)
        self.y_treino = np.array(valores)
        n = len(self.y_treino)
        
        # Construir matriz de distância par-a-par
        print(f"Ajustando Krigagem para {n} pontos...")
        
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_mat[i,j] = np.linalg.norm(self.X_treino[i] - self.X_treino[j])
                
        # Matriz de Covariância K
        K = self._covariancia_func(dist_mat)
        
        # Adicionar restrição de Lagrangiano (Krigagem Ordinária: soma pesos = 1)
        # Sistema estendido:
        # | K   1 | | w |   | k |
        # | 1^T 0 | | mu| = | 1 |
        
        K_ext = np.ones((n+1, n+1))
        K_ext[:n, :n] = K
        K_ext[n, n] = 0
        
        # Inverter K_ext (parte custosa O(N^3))
        # Para N pequeno (<1000) é rápido.
        try:
            self.K_inv = np.linalg.inv(K_ext)
            print("Matriz invertida com sucesso.")
        except np.linalg.LinAlgError:
            print("ERRO: Matriz singular. Verifique pontos duplicados.")
            
    def predizer(self, pontos_alvo):
        """
        Estima valores nos pontos alvo.
        Pontos alvo: array (M, 2)
        """
        pontos_alvo = np.array(pontos_alvo)
        m = len(pontos_alvo)
        n = len(self.y_treino)
        
        estimativas = np.zeros(m)
        variancias = np.zeros(m)
        
        for i in range(m):
            pt = pontos_alvo[i]
            
            # Distâncias para os pontos de treino
            dists = np.linalg.norm(self.X_treino - pt, axis=1)
            
            # Vetor k (covariância alvo-treino)
            k_cov = self._covariancia_func(dists)
            
            # Estender k
            k_ext = np.ones(n+1)
            k_ext[:n] = k_cov
            
            # Pesos lambda = K_inv * k
            pesos = np.dot(self.K_inv, k_ext)
            
            # Valor Estimado = soma(lambda_i * z_i) (descartando o mu do Lagrangiano)
            lambdas = pesos[:n]
            estimativa = np.sum(lambdas * self.y_treino)
            estimativas[i] = estimativa
            
            # Variância de Krigagem (Erro)
            # sigma^2 = Sill - sum(lambda * Cov) - mu
            # mu é o último elemento de 'pesos'
            mu = pesos[n]
            var = self.c0 - np.sum(lambdas * k_cov) - mu
            variancias[i] = var
            
        return estimativas, variancias

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Krigagem Ordinária Manual...")
    
    # 1. Pontos fictícios (Estações Meteorológicas)
    # Coordenadas (Lon, Lat) simplificadas
    coords = np.array([
        [0, 0], [10, 0], [0, 10], [10, 10], # Quadrado
        [5, 5] # Centro
    ])
    
    # Valores (ex: Temperatura)
    # Gradiente Diagonal Quente -> Frio
    vals = np.array([30.0, 25.0, 25.0, 20.0, 25.0]) 
    
    krig = KrigagemSimples(alcance=15.0, patamar=20.0, pepita=0.1)
    krig.ajustar(coords, vals)
    
    # 2. Criar Grade para Mapa
    grid_x = np.linspace(-2, 12, 20)
    grid_y = np.linspace(-2, 12, 20)
    XX, YY = np.meshgrid(grid_x, grid_y)
    
    pontos_grid = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Predizer
    z_pred, z_var = krig.predizer(pontos_grid)
    ZZ = z_pred.reshape(XX.shape)
    
    print("Predição concluída. Gerando mapa...")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(XX, YY, ZZ, levels=20, cmap='coolwarm')
    plt.colorbar(label='Temperatura Estimada (°C)')
    
    # Plotar pontos originais
    plt.scatter(coords[:,0], coords[:,1], c=vals, edgecolor='black', s=100, cmap='coolwarm')
    
    plt.title('Interpolação Espacial (Krigagem Ordinária)\nTeste com 5 Estações')
    plt.xlabel('Longitude Relativa')
    plt.ylabel('Latitude Relativa')
    
    print("Teste finalizado. Sistema de Krigagem funcional.")
