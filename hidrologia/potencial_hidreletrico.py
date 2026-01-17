import numpy as np

"""
MÓDULO DE HIDROLOGIA E ENERGIA: POTENCIAL HIDRELÉTRICO
======================================================

Calcula a potência gerada por uma usina hidrelétrica.
P = rho * g * Q * H * eta

Onde:
rho: densidade água (1000 kg/m3)
g: gravidade (9.81 m/s2)
Q: Vazão turbinada (m3/s)
H: Altura de queda líquida (m)
eta: Eficiência global (turbina + gerador)

Também calcula Energia (MWh).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class CalculadoraHidreletrica:
    def __init__(self, eficiencia=0.92):
        self.eta = eficiencia
        self.rho = 1000.0 # kg/m3
        self.g = 9.81 # m/s2
        
    def calcular_potencia_instatanea_mw(self, vazao_m3s, queda_m):
        """Retorna Potência em Megawatts (MW)."""
        # P (Watts) = 1000 * 9.81 * Q * H * 0.92
        watts = self.rho * self.g * vazao_m3s * queda_m * self.eta
        megawatts = watts / 1e6
        return megawatts

    def calcular_energia_diaria_mwh(self, vazao_media_m3s, queda_m):
        """Retorna Energia gerada em 24h (MWh)."""
        potencia_mw = self.calcular_potencia_instatanea_mw(vazao_media_m3s, queda_m)
        energia_mwh = potencia_mw * 24.0
        return energia_mwh

    def curva_colina(self, vazao, queda_nominal):
        """Simula perda de eficiência fora da vazão nominal (Curva Colina simplificada)."""
        # Eficiência cai se vazão muito baixa ou muito alta
        # Assumindo vazão nominal = 100% da ref
        perda = 1.0
        # Futuro: Implementar curva quadrática de perda
        return self.calcular_potencia_instatanea_mw(vazao, queda_nominal) * perda

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Cálculo Hidrelétrico...")
    
    calc = CalculadoraHidreletrica(eficiencia=0.90)
    
    # Itaipu (Aprox): Queda ~118m
    # Vazão ~10.000 m3/s (por turbina? não, total. 14000 MW total)
    # Digamos uma turbina Francis grande: 700 MW
    
    q = 650.0 # m3/s
    h = 118.0 # m
    
    p = calc.calcular_potencia_instatanea_mw(q, h)
    
    print(f"Turbina (Q={q} m³/s, H={h} m) -> Potência: {p:.2f} MW")
    
    e = calc.calcular_energia_diaria_mwh(q, h)
    print(f"Energia Diária: {e:.2f} MWh")
    print(f"Receita Estimada (R$ 200/MWh): R$ {e * 200:,.2f}")
