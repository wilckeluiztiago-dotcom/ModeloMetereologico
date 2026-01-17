import numpy as np

def calcular_vorticidade_relativa(u, v, dx, dy):
    """
    Calcula a vorticidade relativa (zeta).
    zeta = dv/dx - du/dy
    """
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    
    zeta = dv_dx - du_dy
    return zeta

def adveccao_vorticidade(zeta, u, v, dx, dy):
    """
    Calcula a advecção de vorticidade: - (u * dZeta/dx + v * dZeta/dy)
    """
    dzeta_dx = np.gradient(zeta, dx, axis=1)
    dzeta_dy = np.gradient(zeta, dy, axis=0)
    
    adv = - (u * dzeta_dx + v * dzeta_dy)
    return adv
