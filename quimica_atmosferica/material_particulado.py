import numpy as np

"""
MÓDULO DE QUÍMICA ATMOSFÉRICA: DINÂMICA DE MATERIAL PARTICULADO (PM2.5)
=======================================================================

Modela o balanço de massa de PM2.5 considerando:
- Emissão primária
- Deposição seca (Gravitacional/Impactação)
- Deposição úmida (Washout pela chuva)

dC/dt = E - Vd*C/H - Lambda*C (Chuva)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class DinamicaParticulados:
    def __init__(self, altura_camada_mistura=1000.0):
        self.H = altura_camada_mistura # metros
        self.vd = 0.005 # Velocidade deposição seca (m/s) para PM2.5 (muito lento)
        
    def coeficiente_washout(self, taxa_chuva_mmh):
        """
        Calcula coeficiente de lavagem (Scavenging coefficient) lambda (1/s).
        Empírico: Lambda = A * P^B
        """
        if taxa_chuva_mmh <= 0: return 0.0
        
        # Coeficientes típicos para partículas finas
        A = 1e-4
        B = 0.8
        
        lamb = A * (taxa_chuva_mmh ** B)
        return lamb

    def passo_tempo(self, conc_atual, emissao_kgs_m2, chuva_mmh, dt_seg=3600):
        """
        Evolui a concentração (kg/m3 ou ug/m3).
        """
        # Termo Fonte (Emissão volumétrica)
        # E_vol = E_area / H
        fonte = emissao_kgs_m2 / self.H
        
        # Termo Sumidouro (Deposição Seca)
        # Loss = Vd * C / H
        perda_seca = (self.vd / self.H) * conc_atual
        
        # Termo Sumidouro (Deposição Úmida)
        lamb = self.coeficiente_washout(chuva_mmh)
        perda_umida = lamb * conc_atual
        
        # Derivada Total
        dC = fonte - perda_seca - perda_umida
        
        c_nova = max(0, conc_atual + dC * dt_seg)
        return c_nova

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Dinâmica de PM2.5...")
    
    modelo = DinamicaParticulados(altura_camada_mistura=500)
    
    conc = 50.0 # ug/m3 inicial (poluido)
    chuva = 10.0 # mm/h (chuva forte)
    emissao = 0.0 # parou de emitir
    
    # Simular 1 hora de chuva
    conc_final = modelo.passo_tempo(conc, emissao, chuva, dt_seg=3600)
    
    print(f"Conc Inicial: {conc} ug/m3")
    print(f"Conc Final (após chuva): {conc_final:.2f} ug/m3")
    print(f"Redução: {(1 - conc_final/conc)*100:.1f}%")
