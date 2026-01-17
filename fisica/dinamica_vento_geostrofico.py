import numpy as np

def calcular_vento_geostrofico(pressao, latitude, dx, dy):
    """
    Calcula o vento geostrófico (Ug, Vg) a partir do campo de pressão.
    Ug = - (1 / (rho * f)) * dP/dy
    Vg = (1 / (rho * f)) * dP/dx
    """
    omega = 7.2921e-5
    rho = 1.225
    
    # Calcular parâmetro de Coriolis (f)
    # Latitude média do Sul do Brasil ~ 25-30S (negativo)
    # Mas usaremos magnitude para calculo escalar simplificado
    lat_rad = np.radians(np.abs(latitude))
    f = 2 * omega * np.sin(lat_rad)
    
    dp_dx = np.gradient(pressao, dx, axis=1) * 100 # hPa to Pa
    dp_dy = np.gradient(pressao, dy, axis=0) * 100
    
    ug = - (1 / (rho * f)) * dp_dy
    vg = (1 / (rho * f)) * dp_dx
    
    return ug, vg
