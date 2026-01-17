import numpy as np

def simulacao_onda_rossby_simplificada(latitude, comprimento_onda):
    """
    Calcula a velocidade de fase das ondas de Rossby (Ondas Planetárias).
    c = U - (beta * L^2) / (4 * pi^2)
    
    Onde:
    U = vento médio zonal (Westerly)
    beta = df/dy (variação de Coriolis com latitude)
    """
    omega = 7.2921e-5
    raio_terra = 6.371e6
    lat_rad = np.radians(latitude)
    
    beta = (2 * omega * np.cos(lat_rad)) / raio_terra
    
    u_medio = 10.0 # m/s (Oeste para Leste)
    
    c = u_medio - (beta * comprimento_onda**2) / (4 * np.pi**2)
    return c
