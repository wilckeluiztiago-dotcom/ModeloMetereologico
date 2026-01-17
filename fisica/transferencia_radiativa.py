import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn # Exponencial integral para transferência radiativa

"""
MÓDULO DE TRANSFERÊNCIA RADIATIVA ATMOSFÉRICA
==============================================

Resolve a Equação de Schwarzschild para radiação de Onda Longa (Terrestre)
e Onda Curta (Solar) através de camadas atmosféricas.
Simula o Efeito Estufa e aquecimento solar.

Usa aproximação de Dois Fluxos (Two-Stream Approximation) ou Emissividade de Banda Larga.

Equação Básica: dI/dtau = -I + B(T)
Onde tau é a espessura óptica e B(T) a função de Planck.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloRadiacao:
    def __init__(self, n_camadas=50):
        self.n = n_camadas
        self.sigma = 5.67e-8 # Stefan-Boltzmann
        
    def funcao_planck_banda_larga(self, temp_k):
        """Integração da Lei de Planck: sigma * T^4 / pi."""
        return (self.sigma * temp_k**4) / np.pi

    def calcular_espessura_optica(self, pressao, umidade_especifica, co2_ppm=420):
        """
        Calcula dTau para cada camada baseado em absorvedores (H2O, CO2).
        Tau = k * density * dz
        Simulamos k dependente de P e T (broadening).
        """
        # Simplificação física
        # H2O absorve muito no infravermelho
        # CO2 absorve em bandas específicas
        
        # dTau proporcional a q (umidade) * dp
        k_h2o = 0.1 # Coeficiente de absorção de massa efetivo
        k_co2 = 0.002 * (co2_ppm / 400.0)
        
        # Diferença de pressão como proxy de massa (Lei hidrostática)
        dp = np.abs(np.diff(pressao))
        q_mid = (umidade_especifica[:-1] + umidade_especifica[1:]) / 2
        
        d_tau = (k_h2o * q_mid + k_co2) * dp / 9.81
        return d_tau

    def resolver_schwarzschild_onda_longa(self, pressoes, temperaturas, umidade):
        """
        Calcula os fluxos radiativos ascendente (Upwell) e descendente (Downwell).
        Retorna as taxas de aquecimento/resfriamento (Heating Rates).
        """
        n = len(pressoes)
        d_tau = self.calcular_espessura_optica(pressoes, umidade)
        
        # Fluxos definidos nas interfaces das camadas
        flux_up = np.zeros(n)
        flux_down = np.zeros(n)
        
        # Condições de Contorno
        # Topo da atmosfera: Downwell LW = 0 (espaço frio)
        flux_down[0] = 0 
        # Superfície: Upwell = Emissão corpo negro
        flux_up[-1] = self.sigma * temperaturas[-1]**4
        
        # Função fonte (Planck) nas camadas médias
        B_layer = self.sigma * ((temperaturas[:-1] + temperaturas[1:])/2)**4
        
        # Integração Descendente (Downwards)
        # F_down(i+1) = F_down(i)*exp(-dt) + B*(1-exp(-dt))
        # Usando transmissividade T = exp(-1.66 * dtau) (Fator de difusividade 1.66)
        transmissividade = np.exp(-1.66 * d_tau)
        emissividade = 1.0 - transmissividade
        
        for i in range(n-1):
            flux_down[i+1] = flux_down[i] * transmissividade[i] + B_layer[i] * emissividade[i]
            
        # Integração Ascendente (Upwards)
        # Começa da superfície (índice -1) para cima
        for i in range(n-2, -1, -1):
            flux_up[i] = flux_up[i+1] * transmissividade[i] + B_layer[i] * emissividade[i]
            
        flux_net = flux_up - flux_down
        
        # Taxa de Aquecimento (Heating Rate): dT/dt = - (g/Cp) * dF_net/dp
        g = 9.81
        cp = 1004.0
        dp = np.abs(np.diff(pressoes)) * 100 # hPa -> Pa
        
        # dF_net nas camadas
        df_net = np.diff(flux_net) # F_top - F_bot
        
        heating_rate = - (g / cp) * (df_net / dp)
        heating_rate_k_day = heating_rate * 86400 # K/s -> K/dia
        
        return flux_up, flux_down, heating_rate_k_day

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Solucionador de Transferência Radiativa...")
    
    # Perfil Atmosférico Padrão
    z = np.linspace(20000, 0, 50) # 20km até superfície
    p = 1000 * np.exp(-z/7000) # Perfil exp pressão
    t = 288 - 6.5 * (z/1000) # Troposfera padrão
    t[t < 216] = 216 # Tropopausa
    
    # Perfil de umidade (exponencial decaindo com altura)
    q = 0.015 * np.exp(-z/2000) # 15 g/kg na sup
    
    modelo = ModeloRadiacao()
    up, down, heat = modelo.resolver_schwarzschild_onda_longa(p, t, q)
    
    print("\nResultados Finais:")
    print(f"OLR (Outgoing Longwave Radiation no Topo): {up[0]:.2f} W/m²")
    print(f"Fluxo Descendente na Superfície (Efeito Estufa): {down[-1]:.2f} W/m²")
    
    # Plotar Perfil de Resfriamento Radiativo
    plt.figure(figsize=(6, 8))
    p_layer = (p[:-1] + p[1:])/2
    plt.plot(heat, p_layer, 'r-')
    plt.gca().invert_yaxis()
    plt.xlabel('Taxa de Aquecimento Radiativo (K/dia)')
    plt.ylabel('Pressão (hPa)')
    plt.title('Perfil de Resfriamento Radiativo (Onda Longa)')
    plt.grid(True)
    print("Gráfico radiativo gerado.")
