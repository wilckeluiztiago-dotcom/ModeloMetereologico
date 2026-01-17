import numpy as np

def estimar_velocidade_vertical_omega(vorticidade_adveccao, adveccao_termica_laplaciano):
    """
    Equação Omega Quase-Geostrófica (Simplificada).
    L(omega) proportional to d/dz(Adv_vorticidade) + Laplacian(Adv_termica)
    
    Indica movimento vertical ascendente (mau tempo) ou descendente (bom tempo).
    Valores negativos de Omega implicam subida do ar (chuva/nuvens).
    """
    # Simplificação: Omega é proporcional a soma dos termos forçantes
    # Um termo positivo aqui geralmente "força" subida se os sinais estiverem certos na eq completa
    
    omega_force = -vorticidade_adveccao - adveccao_termica_laplaciano
    return omega_force
