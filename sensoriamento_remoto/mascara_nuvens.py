import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: MÁSCARA DE NUVENS
=================================================

Identifica pixels nublados em imagens de satélite baseada em limiares
térmicos (Banda Infravermelho Térmico) e reflectância (Visível).

Nuvens são frias (Topo alto) e brilhantes.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class MascaraNuvens:
    def __init__(self):
        pass
        
    def gerar_mascara(self, banda_visivel, banda_termica_kelvin, limiar_vis=0.3, limiar_temp=280):
        """
        Retorna máscara Booleana (True = Nuvem).
        Visível: 0-1 (Reflectância)
        Térmica: Kelvin (Brightness Temperature)
        """
        # Critério 1: Brilho (Nuvens refletem muito)
        mask_vis = banda_visivel > limiar_vis
        
        # Critério 2: Frio (Topo da nuvem é gelado)
        mask_temp = banda_termica_kelvin < limiar_temp
        
        # Combinação (Nuvem deve ser CLARA E FRIA)
        # Nuvens baixas podem ser quentes, cirrus podem ser escuras no visível.
        # Mascara simples combinada intersect
        mascara_final = mask_vis & mask_temp
        
        return mascara_final

    def aplicar_mascara(self, imagem, mascara, valor_fill=np.nan):
        """Aplica NaN ou cor onde tem nuvem."""
        img_masked = imagem.copy()
        img_masked[mascara] = valor_fill
        return img_masked

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Testando Máscara de Nuvens...")
    
    # Criar cena sintética
    # Fundo quente e escuro (Terra), Objeto frio e claro (Nuvem)
    vis = np.zeros((100, 100)) + 0.1 # Terra
    vis[40:60, 40:60] = 0.8 # Nuvem quadrada
    
    temp = np.zeros((100, 100)) + 300 # 27C Terra
    temp[40:60, 40:60] = 240 # -33C Nuvem
    
    mascarador = MascaraNuvens()
    mask = mascarador.gerar_mascara(vis, temp)
    
    plt.imshow(mask, cmap='gray')
    plt.title("Máscara de Nuvens Detectada")
    # plt.show()
    
    print(f"Pixels de Nuvem: {np.sum(mask)}")
