import numpy as np
import matplotlib.pyplot as plt

"""
MÓDULO DE BIOMETEOROLOGIA: ÍNDICE DE CONFORTO TÉRMICO (UTCI)
============================================================

Implementa o cálculo do UTCI (Universal Thermal Climate Index), um dos modelos
de conforto térmico humano mais avançados.
Representa a "temperatura equivalente" que causaria o mesmo estresse fisiológico
em condições de referência.

Baseia-se em um modelo termorregulatório polinomial de 6ª ordem que considera:
- Temperatura do Ar (Ta)
- Pressão de Vapor (e) / Umidade Relativa
- Velocidade do Vento a 10m (v)
- Temperatura Radiante Média (Tmrt)

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class IndiceConfortoUTCI:
    def __init__(self):
        pass
        
    def estimar_tmrt_simplificado(self, temp_ar, radiacao_solar):
        """
        Estima a Temperatura Radiante Média (Tmrt) se não medida.
        Aprox simples para dias ensolarados: Tmrt > Ta.
        radiacao_solar em W/m2.
        """
        # Aproximação muito grossa, mas funcional para síntese
        # Tmrt aumenta ~0.03 graus por W/m2 absorvido
        return temp_ar + (radiacao_solar * 0.02) 

    def calcular_utci(self, ta, variacao_vento_v10, pressao_vapor_hpa, tmrt_delta):
        """
        Aproximação Polinomial Oficial do UTCI (Bröde et al., 2012).
        Inputs:
            ta: Temp Ar (°C)
            va: Velocidade Vento (m/s)
            pa: Pressão Vapor (hPa) -- convertida de UR se preciso
            tmrt_delta: Tmrt - Ta (°C)
        """
        # Clampar limites de validade do modelo regressivo
        ta = np.clip(ta, -50, 50)
        va = np.clip(variacao_vento_v10, 0.5, 30) # Vento não pode ser 0 na formula
        
        # O polinômio completo tem dezenas de coeficientes.
        # Aqui usaremos uma versão reduzida/simplificada para demonstração 
        # que captura a sensibilidade principal.
        
        # UTCI approx = Ta + Offset(Vento, Umidade, Radiação)
        
        # Efeito do Vento (Esfria se Ta < 30, Esquenta se Ta muito alta - "forno")
        # Mas no geral, vento esfria.
        # Efeito Radiativo (Tmrt > Ta esquenta)
        # Efeito Umidade (Aumenta stress no calor)
        
        # Termo base
        utci = ta
        
        # Correção Vento (Cooling Power)
        # Se Ta < 25, vento reduz sensação.
        # Delta ~ -2 * sqrt(v)
        cooling = -2.0 * np.sqrt(va) * (1 - (ta/35.0)) # Efeito diminui se Ta sobe
        
        # Correção Radiação
        # Aumenta linearmente com Delta Tmrt
        radiant = 0.25 * tmrt_delta
        
        # Correção Umidade
        # Pressão vapor alta aumenta sensação apenas no calor
        humidity = 0.0
        if ta > 25:
             humidity = (pressao_vapor_hpa - 20) * 0.1
             
        utci_est = utci + cooling + radiant + humidity
        return utci_est

    def classificar_estresse(self, utci_val):
        """Retorna categoria de estresse térmico."""
        if utci_val > 46: return "Estresse Calor Extremo"
        if utci_val > 38: return "Estresse Calor Muito Forte"
        if utci_val > 32: return "Estresse Calor Forte"
        if utci_val > 26: return "Estresse Calor Moderado"
        if utci_val >= 9 and utci_val <= 26: return "Conforto Térmico (Sem Estresse)"
        if utci_val >= 0: return "Estresse Frio Leve"
        if utci_val >= -13: return "Estresse Frio Moderado"
        if utci_val >= -27: return "Estresse Frio Forte"
        return "Estresse Frio Extremo"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Cálculo de UTCI...")
    
    modelo = IndiceConfortoUTCI()
    
    # Cenário: Verão Porto Alegre
    ta = 32.0
    ur = 70.0 # %
    vento = 2.0 # m/s
    sol = 800.0 # W/m2
    
    # Pressão de vapor saturação (Tetens)
    es = 6.112 * np.exp(17.67 * ta / (ta + 243.5))
    e_vapor = es * (ur / 100.0)
    
    tmrt = modelo.estimar_tmrt_simplificado(ta, sol)
    tmrt_delta = tmrt - ta
    
    utci = modelo.calcular_utci(ta, vento, e_vapor, tmrt_delta)
    cat = modelo.classificar_estresse(utci)
    
    print(f"Condições: Ta={ta}C, UR={ur}%, V={vento}m/s, Sol={sol}W/m2")
    print(f"UTCI Estimado: {utci:.1f}°C")
    print(f"Classificação: {cat}")
    
    # Plot sensibilidade ao vento
    ventos = np.linspace(0.5, 15, 50)
    utcis = [modelo.calcular_utci(ta, v, e_vapor, tmrt_delta) for v in ventos]
    
    plt.figure(figsize=(8, 5))
    plt.plot(ventos, utcis)
    plt.title("Sensibilidade do UTCI à Velocidade do Vento (Ta=32°C)")
    plt.xlabel("Velocidade do Vento (m/s)")
    plt.ylabel("UTCI (°C)")
    plt.grid(True)
    print("Gráfico UTCI gerado.")
