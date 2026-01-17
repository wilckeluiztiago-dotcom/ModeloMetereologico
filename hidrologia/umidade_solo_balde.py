import numpy as np

"""
MÓDULO DE HIDROLOGIA: UMIDADE DO SOLO (MODELO DE BALDE / BUCKET MODEL)
======================================================================

Modelo conceitual simples para armazenamento de água no solo.
Funciona como um balde que enche com a chuva e esvazia com evapotranspiração.
Se transbordar -> Runoff.

Estados:
- Capacidade de Campo (Field Capacity)
- Ponto de Murcha (Wilting Point)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloBaldeSolo:
    def __init__(self, capacidade_mm=150.0):
        self.cap = capacidade_mm
        self.armazenamento = 0.5 * capacidade_mm # Começa na metade
        self.runoff_acumulado = 0.0
        
    def atualizar_balanco(self, chuva_mm, etp_mm):
        """
        ETP: Evapotranspiração Potencial.
        ET Real depende da disponibilidade de água.
        """
        # 1. Tentar evaporar
        # Se tem água suficiente, evapora ETP. Se não, evapora o que tem.
        # Função de extração linear
        fator_disponibilidade = self.armazenamento / self.cap
        et_real = etp_mm * fator_disponibilidade # Limitação de água
        
        self.armazenamento -= et_real
        if self.armazenamento < 0: self.armazenamento = 0 # Secou
        
        # 2. Adicionar Chuva
        self.armazenamento += chuva_mm
        
        # 3. Verificar Transbordo (Runoff)
        excesso = 0.0
        if self.armazenamento > self.cap:
            excesso = self.armazenamento - self.cap
            self.armazenamento = self.cap
            
        self.runoff_acumulado += excesso
        
        return self.armazenamento, et_real, excesso

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Bucket Model (Umidade Solo)...")
    
    balde = ModeloBaldeSolo(capacidade_mm=100)
    
    # 10 dias de seca
    print("\nSimulando Seca:")
    for i in range(5):
        s, etr, r = balde.atualizar_balanco(0, 5.0)
        print(f"Dia {i+1}: Solo={s:.1f}mm, ET={etr:.1f}mm")
        
    # Chuva forte
    print("\nSimulando Tempestade:")
    s, etr, r = balde.atualizar_balanco(80.0, 2.0)
    print(f"Chuva 80mm -> Runoff: {r:.1f}mm, Solo={s:.1f}mm")
