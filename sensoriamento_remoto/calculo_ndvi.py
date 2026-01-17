import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE SENSORIAMENTO REMOTO: CÁLCULO DE ÍNDICES DE VEGETAÇÃO (NDVI)
======================================================================

Processa matrizes espectrais (Bandas Vermelha e Infravermelho Próximo)
para calcular o NDVI (Normalized Difference Vegetation Index).
Simula o processamento de imagens de satélite (ex: Landsat, Sentinel, GOES).

Fórmula: NDVI = (NIR - RED) / (NIR + RED)
Intervalo: -1 a +1
- Água: < 0
- Solo exposto: 0 - 0.2
- Vegetação densa (Floresta): > 0.6

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ProcessadorSateliteNDVI:
    def __init__(self):
        pass
        
    def gerar_cena_sintetica(self, dim=256):
        """
        Gera bandas NIR e RED sintéticas simulando uma paisagem:
        - Rio (Baixo NIR, Baixo RED)
        - Floresta (Alto NIR, Baixo RED)
        - Cidade (Médio NIR, Alto RED)
        """
        red = np.zeros((dim, dim))
        nir = np.zeros((dim, dim))
        
        # Fundo: Vegetação (Campo)
        red += 0.1
        nir += 0.4
        
        # Objeto 1: Rio (Faixa diagonal)
        for i in range(dim):
            j = i
            if 0 < j < dim:
                red[i, j:min(j+20, dim)] = 0.05 # Água absorve tudo
                nir[i, j:min(j+20, dim)] = 0.05
        
        # Objeto 2: Floresta Densa (Círculo)
        y, x = np.ogrid[:dim, :dim]
        mask_bloom = (x - dim/2)**2 + (y - dim/2)**2 <= (dim/4)**2
        red[mask_bloom] = 0.05 # Clorofila absorve RED
        nir[mask_bloom] = 0.7  # Estrutura foliar reflete NIR
        
        # Ruído do sensor
        red += np.random.normal(0, 0.01, (dim, dim))
        nir += np.random.normal(0, 0.01, (dim, dim))
        
        return np.clip(red, 0, 1), np.clip(nir, 0, 1)

    def calcular_ndvi(self, banda_red, banda_nir):
        """Calcula o índice NDVI evitando divisão por zero."""
        denominador = banda_nir + banda_red
        # Evitar div/0
        denominador[denominador == 0] = 0.0001
        
        ndvi = (banda_nir - banda_red) / denominador
        return np.clip(ndvi, -1, 1)

    def classificar_cobertura(self, ndvi):
        """Segmenta a imagem baseada em thresholds de NDVI."""
        classes = np.zeros_like(ndvi, dtype=int)
        
        classes[ndvi < 0] = 1 # Água
        classes[(ndvi >= 0) & (ndvi < 0.2)] = 2 # Solo/Urbano
        classes[(ndvi >= 0.2) & (ndvi < 0.5)] = 3 # Vegetação Rasteira
        classes[ndvi >= 0.5] = 4 # Floresta
        return classes

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Processamento de NDVI (Satélite)...")
    
    processador = ProcessadorSateliteNDVI()
    
    # Gerar imagem simulada
    red, nir = processador.gerar_cena_sintetica(dim=200)
    
    ndvi = processador.calcular_ndvi(red, nir)
    
    print(f"NDVI Médio: {np.mean(ndvi):.2f}")
    print(f"NDVI Max (Floresta): {np.max(ndvi):.2f}")
    
    # Visualização
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax0 = axes[0].imshow(red, cmap='gray')
    axes[0].set_title('Banda RED (Simulada)')
    
    ax1 = axes[1].imshow(nir, cmap='gray')
    axes[1].set_title('Banda NIR (Simulada)')
    
    # Mapa de cor NBR (Normalized Burn Ratio) ou similar para vegetação
    # Geralmente RdYlGn (Red-Yellow-Green)
    ax2 = axes[2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_title('Índice NDVI Calculado')
    plt.colorbar(ax2, ax=axes[2], label='NDVI')
    
    print("Imagens de satélite geradas.")
