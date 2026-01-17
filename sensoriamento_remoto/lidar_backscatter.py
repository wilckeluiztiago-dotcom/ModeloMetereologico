import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: LIDAR (BACKSCATTER ATMOSFÉRICO)
===============================================================

Simula o sinal de retorno de um pulso LIDAR (Light Grid Detection and Ranging).
O sinal decai com a distância (1/r^2) e extinção, e aumenta com o backscatter (beta)
de aerossóis e nuvens.

Equação LIDAR simples:
P(r) = C * (beta(r) / r^2) * exp(-2 * integral(alpha(r) dr))

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class SimuladorLidar:
    def __init__(self):
        self.C = 1e5 # Constante do sistema (Potência laser, área telescópio)
        
    def simular_perfil(self, alturas, perfil_aerosol_beta, perfil_extincao_alpha):
        """
        Gera sinal de retorno (Potência vs Altura).
        alturas: array (m).
        """
        sinal = np.zeros_like(alturas)
        transmissividade_acum = 1.0
        
        dr = alturas[1] - alturas[0]
        
        for i, r in enumerate(alturas):
            if r <= 0: continue
            
            beta = perfil_aerosol_beta[i]
            alpha = perfil_extincao_alpha[i]
            
            # Atenuação neste passo (Lei de Beer-Lambert)
            tau_passo = np.exp(-2 * alpha * dr)
            transmissividade_acum *= tau_passo
            
            # Potência Retornada
            # Nota: r^2 geometric loss
            potencia = self.C * (beta / (r**2)) * transmissividade_acum
            sinal[i] = potencia
            
        # Logaritmo para visualização (Range Corrected Signal se quiser)
        return sinal

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Testando Simulador LIDAR...")
    
    sim = SimuladorLidar()
    z = np.linspace(10, 5000, 500)
    
    # Atmosfera padrão (exponencial decrescente) + Camada de nuvem a 2km
    beta = 1e-4 * np.exp(-z/1000) 
    beta[200:220] += 5e-3 # Nuvem
    
    alpha = beta * 50.0 # Razão lidar (LR) constante
    
    sinal = sim.simular_perfil(z, beta, alpha)
    
    plt.figure(figsize=(4, 6))
    plt.plot(np.log10(sinal + 1e-9), z)
    plt.title("Sinal LIDAR Simulado (Log)")
    plt.ylabel("Altura (m)")
    plt.xlabel("Log(Potência)")
    plt.grid(True)
    # plt.show()
    print("Perfil LIDAR gerado.")
