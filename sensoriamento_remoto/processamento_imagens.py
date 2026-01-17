import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE SENSORIAMENTO REMOTO: PROCESSAMENTO DE IMAGENS DE SATÉLITE
====================================================================

Funções básicas para manipulação de matrizes raster (bandas espectrais).
- Normalização
- Realce de contraste (Histogram Stretching)
- Composição RGB
- Filtros espaciais (Convolução)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ProcessadorImagemSatelite:
    def __init__(self):
        pass
        
    def normalizar(self, banda):
        """Normaliza para 0-1."""
        min_val = np.min(banda)
        max_val = np.max(banda)
        if max_val == min_val: return np.zeros_like(banda)
        return (banda - min_val) / (max_val - min_val)

    def realce_contraste_linear(self, banda, percentil_min=2, percentil_max=98):
        """Corte e estiramento de histograma."""
        p_min = np.percentile(banda, percentil_min)
        p_max = np.percentile(banda, percentil_max)
        
        img_clip = np.clip(banda, p_min, p_max)
        img_norm = (img_clip - p_min) / (p_max - p_min)
        return img_norm

    def compor_rgb(self, red, green, blue):
        """
        Cria imagem RGB a partir de bandas separadas.
        Espera entradas normalizadas 0-1.
        """
        rgb = np.dstack((red, green, blue))
        return np.clip(rgb, 0, 1)

    def aplicar_filtro_media(self, banda, tamanho=3):
        """Suavização simples (Blur)."""
        # Nota: Ideal usar scipy.ndimage, mas implementado simples com numpy
        # Versão simplificada sem bordas
        h, w = banda.shape
        nova = banda.copy()
        pad = tamanho // 2
        
        for y in range(pad, h-pad):
            for x in range(pad, w-pad):
                janela = banda[y-pad:y+pad+1, x-pad:x+pad+1]
                nova[y, x] = np.mean(janela)
        return nova

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Processamento de Imagens...")
    
    proc = ProcessadorImagemSatelite()
    
    # Criar bandas sintéticas (Gradiente + Ruído)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x, y)
    
    b_red = np.sin(xx * 5) + np.random.normal(0, 0.1, (100, 100))
    b_green = np.cos(yy * 5) + np.random.normal(0, 0.1, (100, 100))
    b_blue = xx * yy
    
    # Realce
    r_enh = proc.realce_contraste_linear(b_red)
    g_enh = proc.realce_contraste_linear(b_green)
    b_enh = proc.realce_contraste_linear(b_blue)
    
    rgb = proc.compor_rgb(r_enh, g_enh, b_enh)
    
    plt.imshow(rgb)
    plt.title("Composição RGB Sintética")
    # plt.show()
    print("Imagem RGB composta gerada.")
