import numpy as np

def parametrizar_nucleacao(umidade_relativa, aerossois_conc):
    """
    Modelo simples de ativação de CCN (Cloud Condensation Nuclei).
    N_act = C * (S)^(k)
    """
    supersaturacao = np.maximum(0, umidade_relativa - 100.0) / 100.0
    c_param = 100.0 # Exemplo
    k_param = 0.7
    
    n_gotas = c_param * (supersaturacao * 100) ** k_param
    return np.where(supersaturacao > 0, n_gotas, 0)
