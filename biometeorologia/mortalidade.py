import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA: MORTALIDADE E MORBIDADE RELACIONADA AO CLIMA
=======================================================================

Estima o excesso de risco de mortalidade (ER) cardiovascular e respiratória
devido a temperaturas extremas (ondas de calor e frio).
Curva em J ou U típica da epidemiologia ambiental.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloMortalidade:
    def __init__(self):
        # Temperatura de Mínima Mortalidade (MMT) para o Sul do Brasil (~21°C)
        self.mmt = 21.0
        
    def risco_relativo_temperatura(self, temperatura_media):
        """
        Calcula Risco Relativo (RR). RR=1.0 é o basal.
        Baseado em curvas epidemiológicas generalizadas.
        """
        rr = 1.0
        
        # Calor: Aumento ~3% por grau acima do limiar
        if temperatura_media > self.mmt:
            dif = temperatura_media - self.mmt
            rr = 1.0 + 0.03 * dif
            # Exponencial em extremos
            if dif > 8: rr *= 1.2 # Onda calor severa acelera risco
            
        # Frio: Aumento ~1.5% por grau abaixo (efeito mais lento mas persistente)
        elif temperatura_media < self.mmt:
            dif = self.mmt - temperatura_media
            rr = 1.0 + 0.015 * dif
            
        return rr

    def estimar_excesso_obitos(self, populacao, taxa_base_diaria_por_100k, rr):
        """
        Retorna número estimado de óbitos extras atribuíveis ao clima.
        """
        obitos_base = (populacao / 100000.0) * taxa_base_diaria_por_100k
        obitos_cenario = obitos_base * rr
        excesso = obitos_cenario - obitos_base
        return excesso

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Modelo de Mortalidade Climática...")
    
    mod = ModeloMortalidade()
    
    # Onda de Calor Porto Alegre (2024)
    # Pop 1.4M. Taxa base ~ 20/100k/dia (Fictício)
    pop = 1400000
    taxa = 2.0 # 2 por 100k (aprox 28 mortes/dia natural)
    
    temp_extrema = 35.0 # MMT é 21
    
    rr = mod.risco_relativo_temperatura(temp_extrema)
    excesso = mod.estimar_excesso_obitos(pop, taxa, rr)
    
    print(f"Temperatura: {temp_extrema}°C (MMT={mod.mmt})")
    print(f"Risco Relativo: {rr:.2f} (+{(rr-1)*100:.1f}%)")
    print(f"Óbitos Extras Estimados: {excesso:.1f} pessoas/dia")
