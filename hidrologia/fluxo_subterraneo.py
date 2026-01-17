import numpy as np

"""
MÓDULO DE HIDROLOGIA E GEOLOGIA: FLUXO DE ÁGUA SUBTERRÂNEA (LEI DE DARCY)
==========================================================================

Modela o movimento da água em meios porosos (aquíferos).
Fluxo = -K * Gradiente Hidráulico * Área.

Aplicável para estimar recarga de base em rios e poços.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloFluxoSubterraneo:
    def __init__(self, condutividade_hidraulica_m_dia=10.0, porosidade=0.25):
        self.K = condutividade_hidraulica_m_dia # m/dia (ex: Areia média)
        self.n = porosidade # Volume de vazios / Volume total
        
    def calcular_vazao_darcy(self, h1, h2, distancia_L, largura_aquifero_W, espessura_b):
        """
        Retorna Q (m3/dia).
        h1, h2: Cargas hidráulicas (nível do lençol) em dois pontos.
        """
        area_secao = largura_aquifero_W * espessura_b
        gradiente_i = (h1 - h2) / distancia_L
        
        # Lei de Darcy: Q = K * A * i
        q = self.K * area_secao * gradiente_i
        return q

    def velocidade_real_poros(self, h1, h2, distancia_L):
        """
        Velocidade média de avanço de um contaminante.
        v = (K * i) / porosidade
        """
        gradiente_i = (h1 - h2) / distancia_L
        v_darcy = self.K * gradiente_i # Velocidade de fluxo aparente
        v_real = v_darcy / self.n
        return v_real

    def rebaixamento_poco(self, Q_bombeamento, h0, r0, r):
        """
        Equação de Thiem para aquíferos confinados (Regime Permanente).
        Q: Vazão bombeada
        h0: Nível original
        r0: Raio de influência
        r: Distância do poço
        Retorna h(r) -> Nível no ponto r.
        """
        # h(r) = h0 - (Q / (2*pi*T)) * ln(r0/r)
        # T = Transmissividade = K * b
        transmissividade = self.K * 10.0 # Assumindo espessura b=10m fixo aqui para exemplo
        
        termo = (Q_bombeamento / (2 * np.pi * transmissividade)) * np.log(r0 / r)
        h_r = h0 - termo
        return h_r

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Fluxo Subterrâneo (Darcy)...")
    
    modelo = ModeloFluxoSubterraneo(condutividade_hidraulica_m_dia=5.0) # Areia fina
    
    # Exemplo: Fluxo entre dois rios
    h1 = 100.0 # m
    h2 = 98.0  # m
    L = 500.0  # m
    W = 1000.0 # m
    b = 20.0   # m
    
    q = modelo.calcular_vazao_darcy(h1, h2, L, W, b)
    v = modelo.velocidade_real_poros(h1, h2, L)
    
    print(f"Vazão através do Aquífero: {q:.1f} m³/dia")
    print(f"Velocidade do Poluente: {v*100:.2f} cm/dia")
