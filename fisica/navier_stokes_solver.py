import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE DINÂMICA DE FLUIDOS COMPUTACIONAL (CFD)
==================================================

Implementa um solucionador numérico para as Equações de Navier-Stokes (2D Incompressível).
Essencial para modelar o movimento do ar na microescala ou mesoescala dentro do modelo regional.

Método: Projeção de Chorin (Fractional Step Method).
1. Advecção-Difusão (Passo Provisório para Velocidade*)
2. Equação de Poisson para Pressão
3. Correção de Velocidade (Projeção)

Equações:
du/dt + (u.nabla)u = -1/rho * grad(p) + nu * laplaciano(u) + F
div(u) = 0

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class NavierStokesSolver:
    def __init__(self, nx=50, ny=50, lx=10000.0, ly=10000.0, nu=10.0, rho=1.225, dt=1.0):
        """
        Args:
            nx, ny: Pontos de grade.
            lx, ly: Dimensões físicas (metros).
            nu: Viscosidade cinemática (m^2/s).
            rho: Densidade do ar (kg/m^3).
            dt: Passo de tempo (s).
        """
        self.nx = nx
        self.ny = ny
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        self.nu = nu
        self.rho = rho
        self.dt = dt
        
        # Campos (u, v, p)
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        # Termos forçantes (ex: Coriolis na escala maior, aqui simplificado)
        self.fx = np.zeros((ny, nx))
        self.fy = np.zeros((ny, nx))

    def _laplaciano(self, f):
        """Calcula Laplaciano discreto (diferenças finitas centradas)."""
        lap = (np.roll(f, 1, axis=1) - 2*f + np.roll(f, -1, axis=1)) / self.dx**2 + \
              (np.roll(f, 1, axis=0) - 2*f + np.roll(f, -1, axis=0)) / self.dy**2
        return lap

    def _adveccao(self, f, u, v):
        """Termo não-linear de advecção: -(u * df/dx + v * df/dy)."""
        # Upwind ou centrada? Usando centrada simples para demonstração (instável sem viscosidade alta)
        # Melhor usar Upwind de primeira ordem para estabilidade em código simples
        
        # Vento positivo (fluxo da esquerda/baixo) usa vizinho anterior
        df_dx = np.zeros_like(f)
        df_dy = np.zeros_like(f)
        
        # Discretização Upwind manual
        # X direction
        flux_x_pos = (f - np.roll(f, 1, axis=1)) / self.dx
        flux_x_neg = (np.roll(f, -1, axis=1) - f) / self.dx
        df_dx = np.where(u > 0, flux_x_pos, flux_x_neg)
        
        # Y direction
        flux_y_pos = (f - np.roll(f, 1, axis=0)) / self.dy
        flux_y_neg = (np.roll(f, -1, axis=0) - f) / self.dy
        df_dy = np.where(v > 0, flux_y_pos, flux_y_neg)
        
        term = - (u * df_dx + v * df_dy)
        return term

    def resolver_poisson_pressao(self, div_u_star, nit=50):
        """
        Resolve laplaciano(p) = rho/dt * div(u*) usando Jacobi Iterativo.
        Computacionalmente intensivo (simula carga de HPC).
        """
        rhs = (self.rho / self.dt) * div_u_star
        p_new = np.zeros_like(self.p)
        
        for _ in range(nit):
            p_new[1:-1, 1:-1] = (
                (self.p[1:-1, 2:] + self.p[1:-1, :-2]) * self.dy**2 +
                (self.p[2:, 1:-1] + self.p[:-2, 1:-1]) * self.dx**2 -
                rhs[1:-1, 1:-1] * self.dx**2 * self.dy**2
            ) / (2 * (self.dx**2 + self.dy**2))
            
            # Condições de Contorno (Neumann dp/dn = 0 nas paredes)
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            
            self.p = p_new.copy()
            
    def passo_tempo(self):
        """Executa um passo completo do algoritmo de projeção."""
        
        # 1. Passo Provisório (Tentativa de velocidade sem pressão)
        # u* = u + dt * (Advecção + Difusão + Forças)
        adv_u = self._adveccao(self.u, self.u, self.v)
        dif_u = self.nu * self._laplaciano(self.u)
        u_star = self.u + self.dt * (adv_u + dif_u + self.fx)
        
        adv_v = self._adveccao(self.v, self.u, self.v)
        dif_v = self.nu * self._laplaciano(self.v)
        v_star = self.v + self.dt * (adv_v + dif_v + self.fy)
        
        # Condições de Contorno de u*, v* (Paredes fechadas ou periódico?)
        # Vamos assumir periódico para simplificar índices
        
        # 2. Equação de Poisson para Pressão
        # div(u*)
        div_u_star = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1)) / (2*self.dx) + \
                     (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0)) / (2*self.dy)
                     
        self.resolver_poisson_pressao(div_u_star)
        
        # 3. Correção de Velocidade (Projeção)
        # u_new = u* - dt/rho * grad(p)
        dp_dx = (np.roll(self.p, -1, axis=1) - np.roll(self.p, 1, axis=1)) / (2*self.dx)
        dp_dy = (np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)) / (2*self.dy)
        
        self.u = u_star - (self.dt / self.rho) * dp_dx
        self.v = v_star - (self.dt / self.rho) * dp_dy
        
        return self.u, self.v, self.p

# ==============================================================================
# SELF-TEST (SIMULAÇÃO DE CAVITY FLOW OU VENTO REGIONAL)
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Solucionador Navier-Stokes 2D (Modo HPC)...")
    
    # Grade de Mesoscala (50km x 50km)
    solver = NavierStokesSolver(nx=40, ny=40, lx=50000, ly=50000, nu=500.0, dt=10.0)
    
    # Condição Inicial: Vento cisalhado ou vórtice
    Y, X = np.mgrid[0:solver.ny, 0:solver.nx]
    
    # Adicionar uma força (Vento geostrófico impulsionando)
    # Forçar vento oeste no topo, leste em baixo
    solver.fx = np.where(Y > solver.ny/2, 0.05, -0.05) 
    
    print("Iterando no tempo (50 passos)...")
    energia_cinetica = []
    
    for t in range(50):
        if t % 10 == 0: print(f"Passo {t}...")
        u, v, p = solver.passo_tempo()
        kinetic = np.sum(0.5 * solver.rho * (u**2 + v**2))
        energia_cinetica.append(kinetic)
        
    print("Simulação concluída.")
    print(f"Energia Cinética Final: {energia_cinetica[-1]:.2e} Joules")
    
    # Visualização do Campo de Vento
    plt.figure(figsize=(8, 8))
    strm = plt.streamplot(X, Y, u, v, color=np.sqrt(u**2 + v**2), cmap='viridis', density=1.5)
    plt.colorbar(strm.lines, label='Velocidade do Vento (m/s)')
    plt.title('Campo de Vento Simulado (Navier-Stokes)')
    print("Gráfico de fluxo gerado.")
