import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: REFLETIVIDADE DE RADAR (RELAÇÃO Z-R)
====================================================================

Converte refletividade medida pelo radar (dBZ) em taxa de precipitação (R em mm/h).
Relação Marshall-Palmer (Padrão): Z = 200 * R^1.6

dBZ = 10 * log10(Z)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class RadarMeteorologico:
    def __init__(self):
        # Coeficientes Marshall-Palmer (Chuva Estratiforme)
        self.a = 200.0
        self.b = 1.6
        
    def dbz_para_chuva(self, dbz_grid):
        """
        Converte dBZ -> Z -> R (mm/h).
        """
        # Z = 10^(dBZ/10)
        Z = 10 ** (dbz_grid / 10.0)
        
        # Z = a * R^b  =>  R = (Z/a)^(1/b)
        R = (Z / self.a) ** (1.0 / self.b)
        
        return R

    def chuva_para_dbz(self, chuva_mmh):
        """Simulador: Gera dBZ a partir da chuva do modelo."""
        # Evitar log de zero
        chuva_safe = np.maximum(0.001, chuva_mmh)
        
        Z = self.a * (chuva_safe ** self.b)
        dbz = 10 * np.log10(Z)
        
        # Limpar fundo (ruído < 0 dBZ)
        dbz[dbz < 5] = -32 # No Data
        return dbz

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Radar Z-R...")
    radar = RadarMeteorologico()
    
    # Chuva forte (50 mm/h)
    r = 50.0
    dbz = radar.chuva_para_dbz(np.array([r]))
    print(f"Chuva: {r} mm/h -> Refletividade: {dbz[0]:.1f} dBZ (Esperado > 50)")
    
    # Inverso
    r_est = radar.dbz_para_chuva(dbz)
    print(f"Recuperado: {r_est[0]:.1f} mm/h")
