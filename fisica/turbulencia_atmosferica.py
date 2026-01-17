import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE TURBULÊNCIA ATMOSFÉRICA
==================================

Simula a Energia Cinética Turbulenta (TKE - Turbulent Kinetic Energy)
e espectros de potência de turbulência (Kolmogorov).
Essencial para prever rajadas de vento (gusts) e dispersão de poluentes.

Usa o modelo K-Epsilon simplificado 1D.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloTurbulenciaTKE:
    def __init__(self, c_mu=0.09):
        # Constantes do modelo k-epsilon padrão
        self.c_mu = c_mu
        self.sigma_k = 1.0
        self.sigma_eps = 1.3
        self.C1_eps = 1.44
        self.C2_eps = 1.92
        self.von_karman = 0.4
        
    def passo_tke(self, k, eps, u_shear, buoyancy, z_grid, dt):
        """
        Avança um passo temporal na equação de TKE (k).
        dk/dt = Produção_Shear + Produção_Buoyancy - Dissipação + Difusão
        
        Args:
            k (array): TKE atual por nível.
            eps (array): Taxa de dissipação epsilon.
            u_shear (array): Gradiente de vento vertical (du/dz)^2.
            buoyancy (array): Termo de empuxo (g/T * dT/dz).
        """
        # Produção Mecânica (Shear) P = nu_t * (du/dz)^2
        # Viscosidade turbulenta nu_t = C_mu * k^2 / eps
        
        with np.errstate(divide='ignore', invalid='ignore'):
            nu_t = self.c_mu * (k**2) / eps
            nu_t = np.nan_to_num(nu_t, nan=0.1) # Segurança
            
        prod_mech = nu_t * u_shear
        
        # Produção/Destruição por Empuxo (G)
        # G = - nu_t * N^2 ... simplificado: proporcional a buoyancy
        prod_buoy = - nu_t * buoyancy
        
        dissipacao = eps
        
        # Equação de k
        dk_dt = prod_mech + prod_buoy - dissipacao
        
        # Update simples (Euler)
        k_novo = k + dk_dt * dt
        
        # Clip para evitar valores negativos físicos
        k_novo = np.maximum(k_novo, 1e-4)
        
        return k_novo, nu_t

    def gerar_espectro_von_karman(self, velocidade_media, desvio_padrao_u, frequencias):
        """
        Gera o espectro de densidade de potência (PSD) teórico de Von Karman
        para flutuações de vento longitudinal.
        
        S_u(f) = (4 * sigma^2 * L / U) / (1 + 70.8 * (f*L/U)^2)^(5/6)
        """
        L = 100.0 # Escala integral de comprimento (m) típica PBL
        U = velocidade_media
        sigma2 = desvio_padrao_u**2
        
        f_norm = (frequencias * L) / U
        
        numerador = 4 * sigma2 * L / U
        denominador = (1 + 70.8 * f_norm**2)**(5.0/6.0)
        
        S_u = numerador / denominador
        return S_u
        
    def fator_rajada(self, tke):
        """
        Estima o fator de rajada (Gust Factor) baseado na TKE.
        Gust ~ U_mean + 3 * sqrt(k)
        """
        u_flutuacao = np.sqrt(tke)
        return 3.0 * u_flutuacao

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Simulação de Turbulência TKE...")
    
    modelo = ModeloTurbulenciaTKE()
    
    # 1. Simulação 1D Vertical
    z = np.linspace(0, 1000, 100)
    
    # Condições iniciais
    k_init = np.ones_like(z) * 0.5
    eps_init = np.ones_like(z) * 0.01
    
    # Perfil de Vento Logarítmico -> Shear decresce com altura
    # u(z) = ln(z), du/dz = 1/z -> shear = (1/z)^2
    shear = (1.0 / (z+10))**2 * 10 # Fator 10 arbitrário p/ força
    
    # Empuxo neutro
    buoy = np.zeros_like(z)
    
    print("Evoluindo TKE por 100 passos...")
    k_hist = []
    
    k_curr = k_init.copy()
    eps_curr = eps_init.copy()
    
    for t in range(100):
        k_curr, nu_t = modelo.passo_tke(k_curr, eps_curr, shear, buoy, z, dt=1.0)
        # Simplificação: Epsilon fixo ou decaindo levemente
        eps_curr = 0.09 * k_curr**1.5 / 50.0 # Length scale fixa 50m
        k_hist.append(k_curr[10]) # Monitorar nível baixo
        
    # Plot evolução
    plt.figure(figsize=(10, 4))
    plt.plot(k_hist)
    plt.title('Evolução Temporal da Energia Cinética Turbulenta (Nível 10m)')
    plt.xlabel('Passo de Tempo')
    plt.ylabel('TKE (m²/s²)')
    plt.grid(True)
    # plt.show()
    print(f"TKE Final a 10m: {k_curr[10]:.4f}")
    
    # 2. Teste Espectro de Vento (Engenharia Eólica)
    print("\nGerando Espectro de Rajada (Von Karman)...")
    freqs = np.logspace(-3, 1, 100) # 0.001 Hz a 10 Hz
    S_u = modelo.gerar_espectro_von_karman(velocidade_media=10.0, desvio_padrao_u=1.5, frequencias=freqs)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(freqs, freqs * S_u) # Espectro pré-multiplicado por f é padrão em meteo
    # Kolmogorov slope -2/3 line visualization
    plt.plot(freqs, 10 * freqs**(-2/3), 'r--', label='Inclinação Kolmogorov -2/3')
    
    plt.title('Espectro de Potência da Turbulência (Modelo Von Karman)')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('f * S(f)')
    plt.legend()
    print("Gráficos de turbulência gerados.")
