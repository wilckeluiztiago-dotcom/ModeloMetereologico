import numpy as np
import pandas as pd

"""
MÓDULO DE HIDROLOGIA: BALANÇO HÍDRICO CLIMATOLÓGICO (THORNTHWAITE-MATHER)
=========================================================================

Cálculo mensal da contabilidade hídrica:
P - ETP -> Alteração no ARM (Armazenamento) -> ETR (Real) -> DEF (Déficit) -> EXC (Excedente).

Fundamental para "Zoneamento Agroclimático".

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class BalancoHidricoThornthwaite:
    def __init__(self, cad=100.0):
        self.cad = cad # Capacidade de Água Disponível (mm)
        
    def calcular_balanco_mensal(self, precips, temps, lats):
        """
        Série de Precipitacao e Temperatura Mensal.
        Retorna DF com ETP, ETR, DEF, EXC.
        """
        n = len(precips)
        # 1. Calcular ETP (Thornthwaite, 1948) baseada em Temp
        # Índice de Calor Anual I
        i_mensal = (np.array(temps) / 5.0) ** 1.514
        I = np.sum(i_mensal) # Supõe que a série é de 1 ano ou média (simplificação)
        
        # Expoente a
        a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
        
        etp_nao_corr = 16.0 * (10 * np.array(temps) / I) ** a
        
        # Correção por fotoperíodo (latitude) - Ignorado na versão simplificada
        etp = etp_nao_corr # Assumindo 12h sol (equador ou média)
        
        # 2. Balanço Sequencial
        arm = self.cad # Solo cheio no início
        lista_etr = []
        lista_def = []
        lista_exc = []
        
        for p, et_pot in zip(precips, etp):
            p_et = p - et_pot
            neg_acum = 0
            
            alt_arm = 0
            novo_arm = arm
            etr = 0
            exc = 0
            defic = 0
            
            if p_et >= 0:
                # Sobra água
                novo_arm = min(self.cad, arm + p_et)
                etr = et_pot
                exc = max(0, (arm + p_et) - self.cad)
                defic = 0
            else:
                # Falta água (retira do solo exponencialmente)
                # Modelo de decaimento ARM = CAD * exp(Acc_Neg / CAD) - Complexo
                # Usar simplificado linear "Balde"
                retirada = min(arm, abs(p_et))
                novo_arm = arm - retirada
                etr = p + retirada
                defic = et_pot - etr
                exc = 0
                
            arm = novo_arm
            lista_etr.append(etr)
            lista_def.append(defic)
            lista_exc.append(exc)
            
        return pd.DataFrame({
            'P': precips, 'T': temps, 'ETP': etp, 'ETR': lista_etr,
            'DEF': lista_def, 'EXC': lista_exc
        })

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Balanço Hídrico Climatológico...")
    
    bh = BalancoHidricoThornthwaite(cad=100)
    
    # Clima típico RS (Inverno chuvoso/frio, Verão quente/seco as vezes)
    temp = [24, 23, 21, 18, 15, 12, 12, 14, 16, 19, 21, 23]
    pre =  [120, 110, 100, 90, 80, 90, 100, 110, 120, 130, 100, 90] # Bem distribuído
    
    res = bh.calcular_balanco_mensal(pre, temp, -30)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(res.round(1))
    
    print(f"Déficit Hídrico Anual: {res['DEF'].sum():.1f} mm")
    print(f"Excedente Hídrico Anual: {res['EXC'].sum():.1f} mm")
