import numpy as np

"""
MÓDULO DE HIDROLOGIA: ROTEAMENTO DE RIOS (MÉTODO DE MUSKINGUM)
==============================================================

Propaga uma onda de cheia através de um trecho de rio.
Considera efeitos de armazenamento (Wedge e Prism storage).

Equação:
O2 = C0*I2 + C1*I1 + C2*O1
Onde I = Inflow, O = Outflow.

Coeficientes dependem de K (tempo de trânsito) e X (fator de peso).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class RoteamentoMuskingum:
    def __init__(self, k_horas=12.0, x_fator=0.2, dt_horas=1.0):
        self.K = k_horas
        self.X = x_fator
        self.dt = dt_horas
        
        # Cálculo dos coeficientes C0, C1, C2
        # C0 = (-KX + 0.5dt) / D
        # C1 = (KX + 0.5dt) / D
        # C2 = (K - KX - 0.5dt) / D
        # D = (K - KX + 0.5dt)
        
        D = self.K * (1 - self.X) + 0.5 * self.dt
        
        self.C0 = (-self.K * self.X + 0.5 * self.dt) / D
        self.C1 = (self.K * self.X + 0.5 * self.dt) / D
        self.C2 = (self.K * (1 - self.X) - 0.5 * self.dt) / D
        
        # Cheqer soma = 1
        soma = self.C0 + self.C1 + self.C2
        # print(f"Coeficientes: {self.C0:.3f}, {self.C1:.3f}, {self.C2:.3f} (Soma={soma:.3f})")

    def propagar_onda(self, hydrograma_entrada):
        """
        Recebe lista/array de vazões de entrada I(t).
        Retorna O(t).
        """
        n = len(hydrograma_entrada)
        outflow = np.zeros(n)
        
        # Condição inicial: O[0] = I[0] (Fluxo estável inicial)
        outflow[0] = hydrograma_entrada[0]
        
        for t in range(1, n):
            I2 = hydrograma_entrada[t]
            I1 = hydrograma_entrada[t-1]
            O1 = outflow[t-1]
            
            O2 = self.C0 * I2 + self.C1 * I1 + self.C2 * O1
            outflow[t] = max(0, O2)
            
        return outflow

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Testando Roteamento Muskingum...")
    
    # Hidrograma de entrada (Cheia triangular)
    t = np.arange(0, 48, 1) # 48 horas
    inflow = np.zeros_like(t, dtype=float) + 10 # Base flow
    inflow[5:15] = np.linspace(10, 100, 10) # Subida
    inflow[15:25] = np.linspace(100, 10, 10) # Descida
    
    router = RoteamentoMuskingum(k_horas=6.0, x_fator=0.2)
    outflow = router.propagar_onda(inflow)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, inflow, label='Entrada (Montante)')
    plt.plot(t, outflow, label='Saída (Jusante)', linestyle='--')
    plt.title("Amortecimento e Atraso da Onda de Cheia (Muskingum)")
    plt.xlabel("Tempo (horas)")
    plt.ylabel("Vazão (m³/s)")
    plt.legend()
    plt.grid(True)
    
    peak_in = np.max(inflow)
    peak_out = np.max(outflow)
    lag = np.argmax(outflow) - np.argmax(inflow)
    
    print(f"Pico Entrada: {peak_in:.1f}")
    print(f"Pico Saída: {peak_out:.1f} (Atenuação)")
    print(f"Lag de Pico: {lag} horas")
