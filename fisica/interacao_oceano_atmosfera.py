import numpy as np

def acoplamento_termico_oceano(temp_ar, temp_oceano, velocidade_vento):
    """
    Calcula o fluxo de calor sensível entre oceano e atmosfera usando fórmula bulk.
    H = rho * cp * Ch * U * (Ts - Ta)
    """
    rho = 1.225
    cp = 1005
    ch = 1.2e-3 # Coeficiente de transferência (adimensional)
    
    fluxo_calor_sensivel = rho * cp * ch * velocidade_vento * (temp_oceano - temp_ar)
    return fluxo_calor_sensivel

def indice_oni_simulado(ano, mes):
    """
    Retorna um índice ONI (Oceanic Nino Index) simulado para o ano/mês.
    """
    t = ano + (mes/12.0)
    # Ciclo irregular de ~4 anos
    valor = 1.5 * np.sin(2 * np.pi * t / 4) + 0.5 * np.sin(2 * np.pi * t / 11)
    return valor
