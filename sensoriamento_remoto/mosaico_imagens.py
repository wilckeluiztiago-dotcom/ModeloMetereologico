import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: GERAÇÃO DE MOSAICOS
===================================================

Combina múltiplas imagens (tiles) adjacentes em uma única imagem grande.
Trata sobreposição (Overlap) usando média ou corte (Feathering/Blending).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class Mosaicador:
    def __init__(self):
        pass
        
    def criar_mosaico_horizontal(self, img_esquerda, img_direita, overlap_pixels=10):
        """
        Junta duas imagens horizontalmente com blending na área de sobreposição.
        """
        h, w1 = img_esquerda.shape
        h2, w2 = img_direita.shape
        
        if h != h2:
            raise ValueError("Imagens devem ter mesma altura")
            
        largura_total = w1 + w2 - overlap_pixels
        mosaico = np.zeros((h, largura_total))
        
        # Copiar parte esquerda (segura)
        limit_left = w1 - overlap_pixels
        mosaico[:, :limit_left] = img_esquerda[:, :limit_left]
        
        # Copiar parte direita (segura)
        start_right_mosaic = w1
        mosaico[:, start_right_mosaic:] = img_direita[:, overlap_pixels:]
        
        # Blending na zona de overlap
        # Linear weight from 1 to 0
        for i in range(overlap_pixels):
            alpha = 1.0 - (i / overlap_pixels) # Peso da esquerda
            col_mosaic = limit_left + i
            
            val_esq = img_esquerda[:, limit_left + i]
            val_dir = img_direita[:, i]
            
            mosaico[:, col_mosaic] = val_esq * alpha + val_dir * (1 - alpha)
            
        return mosaico

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Testando Mosaico...")
    
    # Criar duas imagens gradiente
    img1 = np.tile(np.linspace(0, 1, 100), (100, 1))
    img2 = np.tile(np.linspace(0.5, 1.5, 100), (100, 1)) # Começa diferente
    
    mos = Mosaicador()
    resultado = mos.criar_mosaico_horizontal(img1, img2, overlap_pixels=20)
    
    plt.imshow(resultado, cmap='viridis')
    plt.title("Mosaico com Blending")
    plt.axvline(80, color='r', linestyle='--') # Inicio overlap
    plt.axvline(100, color='r', linestyle='--') # Fim overlap
    # plt.show()
    print("Mosaico gerado.")
