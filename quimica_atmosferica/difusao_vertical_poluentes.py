import numpy as np

"""
MÓDULO DE QUÍMICA ATMOSFÉRICA: DIFUSÃO VERTICAL (TEORIA K)
==========================================================

Modela o transporte vertical de poluentes na Camada Limite Planetária (PBL).
Usa a equação da difusão de K (Eddy Diffusivity).

dC/dt = d/dz (Kz * dC/dz)

Kz varia com a altura e estabilidade (Perfil O'Brien).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class DifusaoVerticalK:
    def __init__(self, nz=50, altura_topo=2000.0):
        self.nz = nz
        self.H = altura_topo
        self.dz = self.H / (nz - 1)
        self.z_grid = np.linspace(0, self.H, nz)
        
    def perfil_kz(self, u_star, L, h_pbl):
        """
        Gera perfil de Kz (m2/s) para camada convectiva.
        Kz(z) = k * u* * z * (1 - z/h)^2  (Perfil parabólico simplificado)
        k = Von Karman (0.4)
        """
        k_vk = 0.4
        kz = np.zeros_like(self.z_grid)
        
        for i, z in enumerate(self.z_grid):
            if z < h_pbl:
                kz[i] = k_vk * u_star * z * (1 - z/h_pbl)**2
            else:
                kz[i] = 0.1 # Difusão residual acima da PBL (Livre atmosfera)
                
        return np.maximum(0.1, kz) # Evitar zero

    def resolver_passo(self, C, Kz, dt=60.0):
        """
        Resolve difusão 1D por Diferenças Finitas (Esquema Explícito).
        Critério CFL de estabilidade: dt <= dz^2 / (2 * max(Kz))
        """
        C_new = C.copy()
        
        # Coeficiente de difusão de cada camada (interpolação simples média aritm.)
        # Fluxo F_i+1/2 = -K * (C_i+1 - C_i)/dz
        
        for i in range(1, self.nz - 1):
            K_upper = (Kz[i] + Kz[i+1])/2
            K_lower = (Kz[i] + Kz[i-1])/2
            
            flux_up = -K_upper * (C[i+1] - C[i]) / self.dz
            flux_down = -K_lower * (C[i] - C[i-1]) / self.dz
            
            # dC/dt = - dF/dz
            div_flux = (flux_up - flux_down) / self.dz
            
            C_new[i] = C[i] - div_flux * dt
            
        # Condições de Contorno (Fluxo zero no topo e base por enquanto, conservativo)
        C_new[0] = C_new[1] 
        C_new[-1] = C_new[-2]
        
        return C_new

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Testando Difusão Vertical (K-Theory)...")
    
    modelo = DifusaoVerticalK(nz=100, altura_topo=1500)
    
    # Perfil Kz (Convectivo forte)
    kz = modelo.perfil_kz(u_star=0.5, L=-50, h_pbl=1000)
    
    # Condição Inicial: Fonte no chão (Poluição urbana matinal)
    C = np.zeros(100)
    C[0:10] = 100.0 # Alta conc perto do solo
    
    # Simular evolução (mistura vertical ao longo do dia)
    t_max = 3600 * 3 # 3 horas
    dt = 1.0 # s
    steps = int(t_max / dt)
    
    print(f"Simulando {steps} passos de tempo...")
    
    perfis = [C.copy()]
    C_curr = C.copy()
    
    for i in range(steps):
        C_curr = modelo.resolver_passo(C_curr, kz, dt)
        if i % (steps//5) == 0:
            perfis.append(C_curr.copy())
            
    # Plot
    plt.figure(figsize=(6, 8))
    for i, p in enumerate(perfis):
        plt.plot(p, modelo.z_grid, label=f'T ~ {i*0.6}h')
        
    plt.xlabel('Concentração Poluente')
    plt.ylabel('Altura (m)')
    plt.title('Evolução Vertical da Poluição (Difusão Turbulenta)')
    plt.legend()
    plt.axhline(1000, color='k', linestyle='--', label='Topo PBL')
    print("Gráfico Difusão gerado.")
