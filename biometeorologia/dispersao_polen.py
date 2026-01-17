import numpy as np

"""
MÓDULO DE BIOMETEOROLOGIA: DISPERSÃO DE PÓLEN E ALERGENOS
=========================================================

Estima a concentração de pólen no ar baseada na fenologia e condições meteorológicas.
- Vento: Aumenta liberação e transporte.
- Chuva: Remove pólen (Washout).
- Temperatura/Umidade: Influenciam maturação e deiscência da antera.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ModeloPolen:
    def __init__(self, tipo_planta='graminea'):
        self.tipo = tipo_planta
        # Calendário fenológico simplificado (S. S. Brasil)
        if tipo_planta == 'graminea':
            self.inicio_polinizacao = 270 # Outubro (DOY)
            self.fim_polinizacao = 360 # Dezembro
        else: # Ambrosia
            self.inicio_polinizacao = 60 # Março
            self.fim_polinizacao = 120 # Abril
            
    def calcular_liberacao(self, dia_do_ano, temperatura, umidade, velocidade_vento):
        """
        Retorna concentracao potencial (grãos/m3).
        """
        # 1. Checar Janela Fenológica
        in_season = self.inicio_polinizacao <= dia_do_ano <= self.fim_polinizacao
        if not in_season:
            return 0.0
        
        # 2. Modelo de Emissão (Kato et al.)
        # Favorecido por T alta, UR baixa, Vento moderado
        
        fator_t = max(0, (temperatura - 10) / 20.0) # 30C = 1.0
        
        fator_ur = 0.0
        if umidade < 60: fator_ur = 1.0
        elif umidade > 90: fator_ur = 0.1 # Úmido cola o pólen
        else: fator_ur = 1 - (umidade - 60)/30.0
        
        fator_vento = min(1.0, velocidade_vento / 5.0) # Vento ajuda a soltar
        
        emissao_base = 500.0 # pico grãos/m3
        
        # Perfil gaussiano da estação (pico no meio)
        meio_estacao = (self.inicio_polinizacao + self.fim_polinizacao) / 2
        duracao = self.fim_polinizacao - self.inicio_polinizacao
        fator_sazonal = np.exp(-((dia_do_ano - meio_estacao)**2) / (2 * (duracao/6)**2))
        
        concentracao = emissao_base * fator_sazonal * fator_t * fator_ur * fator_vento
        return concentracao

    def efeito_chuva(self, concentracao_inicial, chuva_mm):
        """Washout."""
        if chuva_mm > 5.0: return concentracao_inicial * 0.1 # Limpou tudo
        if chuva_mm > 1.0: return concentracao_inicial * 0.5
        return concentracao_inicial

    def nivel_alerta(self, concentracao):
        if concentracao < 10: return "Baixo"
        if concentracao < 50: return "Médio"
        if concentracao < 200: return "Alto"
        return "Muito Alto (Alérgicos evitar externo)"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Dispersão de Pólen...")
    
    grama = ModeloPolen('graminea')
    
    # Dia de Primavera (Outubro), Quente, Seco, Ventoso
    dia = 300 # Fim Outubro
    t = 28
    ur = 40
    v = 6.0
    chuva = 0
    
    conc = grama.calcular_liberacao(dia, t, ur, v)
    conc = grama.efeito_chuva(conc, chuva)
    nivel = grama.nivel_alerta(conc)
    
    print(f"Dia {dia}: T={t}C, V={v}m/s -> Pólen: {conc:.1f} grãos/m³")
    print(f"Alerta: {nivel}")
