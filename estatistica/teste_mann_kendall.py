import numpy as np
from scipy.stats import norm

def teste_mann_kendall(x, alpha=0.05):
    """
    Realiza o teste de Mann-Kendall para tendência.
    Retorna: tendência (bool), p-valor, declive de Sen.
    """
    n = len(x)
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])
            
    # Variância
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
        
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha/2)
    
    return h, p, s
