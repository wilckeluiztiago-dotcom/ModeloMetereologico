import numpy as np
import pandas as pd

"""
MÓDULO DE QUÍMICA ATMOSFÉRICA: EMISSÕES VEICULARES LINEARES
===========================================================

Gera inventário de emissões (CO, NOx, HC, PM) baseado em fluxo de tráfego
em rodovias ou malha urbana.

Fatores de emissão baseados em curvas típicas (CETESB/EPA).
Considera frota (Leve, Pesada, Moto).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class InventarioVeicular:
    def __init__(self):
        # Fatores de emissão médios (g/km)
        self.fatores = {
            'leve': {'CO': 0.3, 'NOx': 0.08, 'PM': 0.002, 'HC': 0.05},
            'pesado': {'CO': 1.5, 'NOx': 4.0, 'PM': 0.15, 'HC': 0.3}, # Diesel sujo
            'moto': {'CO': 0.8, 'NOx': 0.05, 'PM': 0.01, 'HC': 0.15}
        }
        
    def calcular_emissao_segmento(self, comprimento_km, fluxo_veiculos_hora):
        """
        Calcula emissão total num trecho de estrada (g/h).
        fluxo_veiculos_hora: dict {'leve': 1000, 'pesado': 200...}
        """
        emissoes_totais = {'CO': 0.0, 'NOx': 0.0, 'PM': 0.0, 'HC': 0.0}
        
        for tipo, qtd in fluxo_veiculos_hora.items():
            if tipo in self.fatores:
                ef = self.fatores[tipo]
                # E = N * L * EF
                for pol, fator in ef.items():
                    carga = qtd * comprimento_km * fator
                    emissoes_totais[pol] += carga
                    
        return emissoes_totais

    def gerar_perfil_horario(self):
        """Gera perfil típico de tráfego (rush hour)."""
        horas = np.arange(24)
        # Pico duplo (Manhã 8h, Tarde 18h)
        trafego = 100 + 500 * np.exp(-((horas-8)**2)/4) + 600 * np.exp(-((horas-18)**2)/4)
        return horas, trafego

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Inventário Veicular...")
    
    inv = InventarioVeicular()
    
    # Rodovia movimentada: 10km
    fluxo = {'leve': 2000, 'pesado': 500, 'moto': 200}
    comp = 10.0
    
    em = inv.calcular_emissao_segmento(comp, fluxo)
    
    print(f"Emissões Totais no Trecho ({comp}km):")
    for p, v in em.items():
        print(f"  {p}: {v/1000:.2f} kg/h") # Converter para kg
