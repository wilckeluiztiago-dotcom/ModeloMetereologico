import numpy as np

def calcular_difusao_umidade(umidade_especifica, k_difusao, dt, dx, dy):
    """
    Resolve a equação de difusão para umidade: dq/dt = K * (d2q/dx2 + d2q/dy2)
    
    Args:
        umidade_especifica (array): Matriz de umidade.
        k_difusao (float): Coeficiente de difusividade.
    """
    d2q_dx2 = np.gradient(np.gradient(umidade_especifica, dx, axis=1), dx, axis=1)
    d2q_dy2 = np.gradient(np.gradient(umidade_especifica, dy, axis=0), dy, axis=0)
    
    laplaciano = d2q_dx2 + d2q_dy2
    
    delta_umidade = k_difusao * laplaciano * dt
    return delta_umidade
