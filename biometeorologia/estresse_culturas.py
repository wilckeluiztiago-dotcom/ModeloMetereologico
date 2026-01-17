import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA AGRÍCOLA: ESTRESSE TÉRMICO EM CULTURAS
================================================================

Analisa impacto de temperaturas extremas em culturas chave do Sul (Soja, Trigo, Milho).
Calcula:
1. Graus-Dia Acumulados (GDD) para fenologia.
2. Dias com Heat Stress (Temperatura > limiar crítico).
3. Penalidade de Produtividade (Simplificada).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloCulturas:
    def __init__(self, cultura='soja'):
        self.cultura = cultura
        if cultura == 'soja':
            self.t_base = 10.0
            self.t_otima_min = 20.0
            self.t_otima_max = 30.0
            self.t_critica = 35.0 # Abortamento de flores
        elif cultura == 'trigo':
            self.t_base = 0.0
            self.t_otima_min = 15.0
            self.t_otima_max = 24.0
            self.t_critica = 30.0 # Choque térmico no enchimento de grãos

    def calcular_gdd(self, t_max, t_min):
        """Growing Degree Days (Graus-Dia)."""
        t_media = (t_max + t_min) / 2
        # Ajuste: se Tmean < Tbase, GDD = 0
        gdd = max(0, t_media - self.t_base)
        return gdd

    def verificar_estresse(self, t_max):
        """Retorna intensidade do estresse [0-1]."""
        if t_max < self.t_critica:
            return 0.0
        
        # Estresse cresce linearmente acima do crítico
        excess = t_max - self.t_critica
        return min(1.0, excess / 5.0) # 5 graus acima = 100% dano

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Estresse em Culturas...")
    
    soja = ModeloCulturas('soja')
    trigo = ModeloCulturas('trigo')
    
    # Verão Quente (38C)
    dano_soja = soja.verificar_estresse(38)
    dano_trigo = trigo.verificar_estresse(38)
    
    print(f"Calor de 38°C:")
    print(f"  Dano Soja (>35): {dano_soja*100:.1f}%")
    print(f"  Dano Trigo (>30): {dano_trigo*100:.1f}%")
    
    # GDD
    gdd = soja.calcular_gdd(30, 20) # Média 25 (Tbase 10) -> 15 GDD
    print(f"GDD Soja (dia ideal): {gdd} graus-dia")
