import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA: PHYSIOLOGICAL EQUIVALENT TEMPERATURE (PET)
=====================================================================

O PET é definido como a temperatura do ar em um ambiente de referência interno típico 
(sem vento e sem radiação solar) que causaria o mesmo balanço térmico no corpo humano.

Simplificação do modelo MEMI (Munich Energy-balance Model for Individuals).
PET ~ Ta + Correc(Vento, Radiação, Umidade).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class IndicePET:
    def __init__(self):
        pass
        
    def estimar_pet_simplificado(self, ta, vapor_pressure_hpa, vento_v12, radiacao_solar):
        """
        Aproximação regressiva para PET.
        Baseado em Walther & Matzarakis (2006) ou similar.
        """
        # PET aumenta drasticamente com o sol e diminui com o vento.
        
        # 1. Efeito Radiativo (Tmrt)
        # Aproximadamente: Tmrt sobe 1C a cada ~30 W/m2 absorvido
        tmrt_proxy = ta + (radiacao_solar * 0.03) 
        
        # 2. Correção Vento (reduz sensação em geral, exceto muito quente)
        # Cooling effect ~ sqrt(v)
        cooling = -3.0 * (vento_v12 ** 0.5)
        
        # 3. Componente Latente (Umidade)
        # Sensação aumenta com vapor d'água
        humidity_stress = (vapor_pressure_hpa - 12) * 0.4
        
        pet_estimado = ta + (tmrt_proxy - ta)*0.6 + cooling + humidity_stress
        return pet_estimado

    def classificar_conforto(self, pet):
        if pet > 41: return "Estresse Calor Extremo"
        if pet > 35: return "Estresse Calor Forte"
        if pet > 29: return "Estresse Calor Moderado"
        if pet > 23: return "Leve Estresse Calor"
        if pet >= 18: return "Confortável"
        if pet >= 13: return "Leve Estresse Frio"
        if pet >= 8: return "Estresse Frio Moderado"
        if pet >= 4: return "Estresse Frio Forte"
        return "Estresse Frio Extremo"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Índice PET...")
    pet_calc = IndicePET()
    
    # Parque em dia de sol
    ta = 30
    sol = 900 # W/m2
    vento = 1.0
    vp = 20 # hPa
    
    pet = pet_calc.estimar_pet_simplificado(ta, vp, vento, sol)
    cat = pet_calc.classificar_conforto(pet)
    
    print(f"Ambiente: Ta={ta}C, Sol={sol}W/m2")
    print(f"PET Estimado: {pet:.1f} °C")
    print(f"Sensação: {cat}")
