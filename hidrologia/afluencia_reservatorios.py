import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
MÓDULO DE HIDROLOGIA E ENERGIA: AFLUÊNCIA DE RESERVATÓRIOS
==========================================================

Modela a transformação de chuva na bacia em vazão afluente ao reservatório
de uma usina hidrelétrica (ex: Itaipu, Machadinho).

Modelo Chuva-Vazão do tipo "Soil Moisture Accounting" (SMA) simplificado.
Considera:
- Interceptação vegetal
- Infiltração no solo
- Escoamento superficial (Runoff)
- Escoamento de base (Groundwater)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloChuvaVazaoReservatorio:
    def __init__(self, area_bacia_km2=1000, capacidade_solo_mm=100):
        self.area = area_bacia_km2
        self.s_max = capacidade_solo_mm # Capacidade de campo
        self.s_atual = 0.5 * self.s_max # Estado inicial (50% úmido)
        
        # Parâmetros
        self.ks = 0.2 # Condutividade hidráulica (infiltração)
        self.k_base = 0.05 # Recessão fluxo de base
        self.k_sup = 0.6 # Tempo resposta escoamento superficial
        
    def passo_tempo(self, precipitacao_mm, evapotranspiracao_mm):
        """
        Executa um passo de tempo diário.
        Retorna Vazão Total (m³/s).
        """
        # 1. Balanço no Solo
        # P_efetiva = Chuva - Evap
        entrada_liquida = precipitacao_mm - evapotranspiracao_mm
        
        escoamento_sup = 0.0
        recarga = 0.0
        
        if entrada_liquida > 0:
            # Se solo saturar, gera runoff
            if self.s_atual + entrada_liquida > self.s_max:
                excesso = (self.s_atual + entrada_liquida) - self.s_max
                self.s_atual = self.s_max
                escoamento_sup = excesso
            else:
                # Parte infiltra, parte escoa direto (função não linear da saturação)
                fator_sat = (self.s_atual / self.s_max) ** 2
                escoamento_sup = entrada_liquida * fator_sat
                self.s_atual += entrada_liquida * (1 - fator_sat)
        else:
            # Secando solo (min 0)
            self.s_atual = max(0, self.s_atual + entrada_liquida)
            
        # 2. Fluxo de Base (Drenagem lenta do aquífero alimentado pelo solo)
        fluxo_base_mm = self.s_atual * self.k_base
        self.s_atual -= fluxo_base_mm # Drenou
        
        # 3. Conversão mm/dia -> m³/s
        # Q = H(mm) * Area(km²) * 1000 / 86400
        fator_conv = (self.area * 1000) / 86400.0
        
        q_sup = escoamento_sup * fator_conv
        q_base = fluxo_base_mm * fator_conv
        
        q_total = q_sup + q_base
        return q_total, self.s_atual

    def simular_serie(self, serie_chuva, serie_evap):
        """Roda para uma série temporal inteira."""
        vazoes = []
        niveis_solo = []
        
        for p, e in zip(serie_chuva, serie_evap):
            q, s = self.passo_tempo(p, e)
            vazoes.append(q)
            niveis_solo.append(s)
            
        return np.array(vazoes), np.array(niveis_solo)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Modelo Chuva-Vazão de Reservatório...")
    
    # Simular 100 dias
    dias = np.arange(100)
    # Chuva pulsada
    chuva = np.zeros(100)
    chuva[10:15] = 50 # Tempestade de 5 dias
    chuva[50:52] = 20
    
    evap = np.ones(100) * 3 # 3mm/dia constante
    
    modelo = ModeloChuvaVazaoReservatorio(area_bacia_km2=2000)
    vazoes, solo = modelo.simular_serie(chuva, evap)
    
    print(f"Pico de Vazão: {np.max(vazoes):.2f} m³/s")
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.bar(dias, chuva, color='blue', alpha=0.3, label='Chuva (mm)')
    ax1.set_ylabel('Precipitação (mm)', color='blue')
    ax1.set_ylim(0, 100)
    ax1.invert_yaxis() # Chuva de cima para baixo
    
    ax2 = ax1.twinx()
    ax2.plot(dias, vazoes, 'k-', linewidth=2, label='Afluência (m³/s)')
    ax2.set_ylabel('Vazão (m³/s)', color='black')
    ax2.set_ylim(0, max(vazoes)*1.5)
    
    plt.title("Hidrograma Simulado: Resposta de Afluência à Chuva")
    print("Gráfico Hidrograma gerado.")
