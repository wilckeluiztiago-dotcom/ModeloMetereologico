import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
MÓDULO DE TEORIA DO CAOS E SISTEMAS DINÂMICOS
=============================================

Investiga a previsibilidade da atmosfera usando o sistema clássico de Lorenz (1963),
que demonstra a dependência sensível das condições iniciais ("Efeito Borboleta").

Calcula expoentes de Lyapunov para quantificar o horizonte de previsibilidade.

Equações de Lorenz:
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AnalisadorCaosLorenz:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        """
        Args:
            sigma: Número de Prandtl (viscosidade/condutividade térmica).
            rho: Número de Rayleigh (diferença de temp superfície/topo).
            beta: Fator geométrico da célula de convecção.
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def _sistema_lorenz(self, estado, t):
        x, y, z = estado
        
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        
        return [dxdt, dydt, dzdt]

    def simular_trajetoria(self, estado_inicial, t_max=100.0, passos=10000):
        """Integra o sistema no tempo."""
        t = np.linspace(0, t_max, passos)
        solucao = odeint(self._sistema_lorenz, estado_inicial, t)
        return t, solucao

    def calcular_divergencia_trajetorias(self, est_ini_1, perturbacao=1e-5, t_max=50.0):
        """
        Simula duas trajetórias muito próximas para ver a divergência.
        Demonstra o limite da previsão do tempo.
        """
        est_ini_2 = np.array(est_ini_1) + perturbacao
        
        t, sol1 = self.simular_trajetoria(est_ini_1, t_max, passos=5000)
        _, sol2 = self.simular_trajetoria(est_ini_2, t_max, passos=5000)
        
        # Distância Euclideana entre os estados no tempo
        distancia = np.linalg.norm(sol1 - sol2, axis=1)
        
        return t, distancia

    def estimar_expoente_lyapunov_local(self, distancia_tempo, t):
        """
        Lyapunov Exponent (lambda): |delta_Z(t)| ~ e^(lambda * t) |delta_Z0|
        log(dist) ~ lambda * t + C
        Calcula a inclinação da reta log(dist) vs t na fase de crescimento exponencial linear.
        """
        # Pegar apenas a parte onde a distância ainda é pequena (antes da saturação no atrator)
        # Heurística: dist < 5.0
        mask = (distancia_tempo > 1e-4) & (distancia_tempo < 5.0)
        
        if np.sum(mask) < 10:
            return 0.0 # Falha na estimativa
            
        logs = np.log(distancia_tempo[mask])
        tempos = t[mask]
        
        coefs = np.polyfit(tempos, logs, 1)
        lambda_max = coefs[0]
        
        return lambda_max

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Análise de Caos (Lorenz 1963)...")
    
    lorenz = AnalisadorCaosLorenz()
    estado_base = [1.0, 1.0, 1.0]
    
    # 1. Simulação Simples (O Atrator)
    t, traj = lorenz.simular_trajetoria(estado_base)
    
    # Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], lw=0.5, color='darkblue')
    ax.set_title("Atrator de Lorenz (Caos Determinístico)")
    ax.set_xlabel("X (Convecção)")
    ax.set_ylabel("Y (Diferença T Horiz)")
    ax.set_zlabel("Z (Diferença T Vert)")
    print("Atrator gerado.")
    
    # 2. Efeito Borboleta
    print("\nCalculando Divergência (Efeito Borboleta)...")
    t_div, dists = lorenz.calcular_divergencia_trajetorias(estado_base, perturbacao=1e-8)
    
    lyap = lorenz.estimar_expoente_lyapunov_local(dists, t_div)
    horizonte_previsao = 1/lyap if lyap > 0 else np.inf
    
    print(f"Expoente de Lyapunov Máximo estimado: {lyap:.4f}")
    print(f"Horizonte de Previsibilidade (~1/lambda): {horizonte_previsao:.2f} unidades de tempo")
    
    # Plot Erro
    plt.figure(figsize=(10, 4))
    plt.semilogy(t_div, dists)
    plt.title(f"Crescimento do Erro Inicial (1e-8) - Efeito Borboleta")
    plt.xlabel("Tempo")
    plt.ylabel("Distância Log (Erro)")
    plt.grid(True)
