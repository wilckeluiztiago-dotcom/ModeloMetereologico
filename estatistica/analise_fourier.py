import numpy as np
from scipy.fft import fft, fftfreq

def analise_espectral_fourier(dados, frequencia_amostragem=1.0):
    """
    Realiza Transformada Rápida de Fourier (FFT) para identificar periodicidades.
    Retorna frequências e amplitudes.
    """
    n = len(dados)
    # Remover média para evitar pico em zero
    dados_detrended = dados - np.mean(dados)
    
    yf = fft(dados_detrended)
    xf = fftfreq(n, 1 / frequencia_amostragem)[:n//2]
    
    amplitudes = 2.0/n * np.abs(yf[0:n//2])
    
    return xf, amplitudes
