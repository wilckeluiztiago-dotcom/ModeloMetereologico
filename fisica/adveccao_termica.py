import numpy as np

def calcular_adveccao_termica(temperatura, vento_u, vento_v, dx, dy):
    """
    Calcula a advecção de temperatura: - (u * dT/dx + v * dT/dy)
    Equação diferencial parcial discretizada.
    
    Args:
        temperatura (array): Campo de temperatura 2D.
        vento_u (array): Componente u do vento (Oeste-Leste).
        vento_v (array): Componente v do vento (Sul-Norte).
        dx, dy (float): Espaçamento da grade.
    
    Return:
        adv (array): Taxa de mudança de temperatura por advecção.
    """
    grad_x = np.gradient(temperatura, dx, axis=1)
    grad_y = np.gradient(temperatura, dy, axis=0)
    
    adveccao = - (vento_u * grad_x + vento_v * grad_y)
    return adveccao
