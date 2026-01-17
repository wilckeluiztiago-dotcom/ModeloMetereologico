import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: CORREÇÃO ATMOSFÉRICA (DOS - DARK OBJECT SUBTRACTION)
====================================================================================

Remove o efeito de espalhamento atmosférico (Path Radiance) que deixa a imagem azulada/brilhante.
Método DOS (Dark Object Subtraction): Assume que pixels escuros (água profunda/sombra) deveriam ser zero.
O valor mínimo encontrado na banda é considerado "névoa" e subtraído de tudo.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class CorrecaoAtmosferica:
    def __init__(self):
        pass
        
    def aplicar_dos(self, banda):
        """Dark Object Subtraction."""
        # Encontra o valor mínimo representativo (histograma 1%)
        # Evita ruído zero absoluto
        path_radiance = np.percentile(banda, 0.1)
        
        banda_corr = banda - path_radiance
        return np.maximum(0, banda_corr) # Clip negativo

    def corrigir_rayleigh(self, banda_azul, angulo_solar_zenital):
        """Correção física simples para espalhamento Rayleigh (Azul)."""
        # Rayleigh decresce com lambda^4
        # Aumenta com caminho óptico (1/cos(theta))
        fator = 0.05 / np.cos(np.radians(angulo_solar_zenital))
        
        banda_corr = banda_azul - fator
        return np.maximum(0, banda_corr)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Correção Atmosférica...")
    corr = CorrecaoAtmosferica()
    
    # Imagem com névoa (Offset aditivo)
    img = np.random.rand(10, 10) + 0.2
    
    img_limpa = corr.aplicar_dos(img)
    
    print(f"Média Original: {np.mean(img):.3f}")
    print(f"Média Corrigida (DOS): {np.mean(img_limpa):.3f}")
