import numpy as np
from scipy import signal

def analise_wavelet_morlet(dados):
    """
    Realiza Transformada Contínua de Wavelet (CWT) usando a wavelet de Ricker (Mexican Hat)
    como aproximação simplificada sem dependência pesada de pywt se não estiver disponível.
    (Aqui usaremos scipy.signal.cwt com ricker para robustez padrão)
    """
    # Larguras para a wavelet (escalas)
    larguras = np.arange(1, 100)
    
    cwtmatr = signal.cwt(dados, signal.ricker, larguras)
    return larguras, cwtmatr
