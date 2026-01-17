import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE HIDROLOGIA E ENERGIA: DEMANDA ENERGÉTICA E CLIMA
==========================================================

Modela a relação não-linear entre Temperatura do Ar e Consumo de Energia Elétrica.
Curva em "U":
- Alta demanda no frio (Chuveiro elétrico/Aquecedor)
- Alta demanda no calor (Ar condicionado)
- Mínima no conforto térmico (~18-22C)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloDemandaEnergia:
    def __init__(self, carga_base_mw=5000):
        self.carga_base = carga_base_mw
        
    def prever_demanda(self, temperatura, hora_dia):
        """
        Retorna Demanda em MW.
        Considera T e ciclo diário (horário de pico).
        """
        # 1. Componente Climática (U-shape)
        # Fator Frio: T < 18 (Sobe quadraticamente)
        fator_frio = 0.0
        if temperatura < 18:
            fator_frio = 20 * (18 - temperatura)**1.5
            
        # Fator Calor: T > 24 (Sobe exponencialmente/quadrático)
        fator_calor = 0.0
        if temperatura > 24:
            fator_calor = 30 * (temperatura - 24)**1.6
            
        impacto_clima = fator_frio + fator_calor
        
        # 2. Componente Horária (Ciclo humano)
        # Pico as 19h, Vale as 4h
        fator_hora = np.sin((hora_dia - 9) * np.pi / 12)**2 
        # Ajuste para ter vale ~0.6 da base e pico ~1.4
        perfil_hora = 0.7 + 0.6 * fator_hora 
        
        demanda_total = (self.carga_base + impacto_clima) * perfil_hora
        return demanda_total

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Demanda Energética vs Temperatura...")
    
    modelo = ModeloDemandaEnergia(carga_base_mw=10000) # Sistema grande
    
    # Varrer faixa de temperatura
    temps = np.linspace(-5, 40, 50)
    demandas = [modelo.prever_demanda(t, hora_dia=19) for t in temps] # Pico
    
    plt.figure(figsize=(8, 5))
    plt.plot(temps, demandas, 'r-', linewidth=2)
    plt.title("Curva de Carga Térmica (Horário de Pico)")
    plt.xlabel("Temperatura Média (°C)")
    plt.ylabel("Demanda de Energia (MW)")
    plt.grid(True)
    plt.axvline(18, linestyle='--', color='green', alpha=0.5, label='Conforto')
    plt.axvline(24, linestyle='--', color='green', alpha=0.5)
    plt.legend()
    # plt.show()
    
    print(f"Minima Demanda (~20C): {min(demandas):.0f} MW")
    print(f"Maxima Demanda (40C): {max(demandas):.0f} MW")
    print("Gráfico Carga gerado.")
