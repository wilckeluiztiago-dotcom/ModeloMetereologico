import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: EXTRAÇÃO DE FEATURES DE IMAGEM
==============================================================

Extrai características de textura e estatísticas de imagens.
Usa GLCM (Gray Level Co-occurrence Matrix) simplificada para textura.
(Contraste, Homogeneidade, Entropia).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ExtratorFeatures:
    def __init__(self):
        pass
        
    def estatisticas_basicas(self, janela):
        """Média, Desvio, Max, Min."""
        return {
            'media': np.mean(janela),
            'std': np.std(janela),
            'max': np.max(janela),
            'min': np.min(janela)
        }

    def textura_glcm_simples(self, janela, niveis=8):
        """
        Calcula Contraste em vizinhança horizontal.
        Quantiza a imagem para 'niveis' tons de cinza primeiro.
        """
        # Quantização
        min_v, max_v = np.min(janela), np.max(janela)
        if max_v == min_v: return {'contraste': 0.0, 'homogeneidade': 1.0}
        
        img_q = ((janela - min_v) / (max_v - min_v) * (niveis - 1)).astype(int)
        
        # GLCM (só horizontal passo 1)
        glcm = np.zeros((niveis, niveis))
        h, w = img_q.shape
        
        for y in range(h):
            for x in range(w - 1):
                i = img_q[y, x]
                j = img_q[y, x+1]
                glcm[i, j] += 1
                
        # Normalizar
        soma = np.sum(glcm)
        if soma == 0: return {'contraste': 0, 'homogeneidade': 1}
        glcm /= soma
        
        # Features
        contraste = 0.0
        homogeneidade = 0.0
        
        for i in range(niveis):
            for j in range(niveis):
                val = glcm[i, j]
                contraste += val * ((i - j) ** 2)
                homogeneidade += val / (1 + abs(i - j))
                
        return {'contraste': contraste, 'homogeneidade': homogeneidade}

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Extração de Features...")
    
    ext = ExtratorFeatures()
    
    # Textura lisa
    img_lisa = np.zeros((10, 10))
    feat_lisa = ext.textura_glcm_simples(img_lisa)
    print(f"Lisa: {feat_lisa}")
    
    # Textura rugosa (Xadrez)
    img_rugosa = np.indices((10, 10)).sum(axis=0) % 2
    feat_rugosa = ext.textura_glcm_simples(img_rugosa)
    print(f"Rugosa: {feat_rugosa}")
