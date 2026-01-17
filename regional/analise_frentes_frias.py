import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE ANÁLISE DE FRENTES FRIAS
==================================

O Rio Grande do Sul e Santa Catarina são as portas de entrada das frentes frias 
polares no Brasil. Este módulo identifica a passagem desses sistemas baseando-se
em mudanças bruscas de vento (rotação N -> S/SW), queda de temperatura e 
variação de pressão.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AnalisadorFrentesFrias:
    def __init__(self):
        self.criterios = {
            'queda_temp_min': 4.0, # Graus em 24h
            'aumento_pressao_min': 4.0, # hPa em 24h (pós-frente)
            'mudanca_vento': True # Se requer rotação
        }
        
    def identificar_passagens(self, df):
        """
        Identifica datas de provável passagem de frente fria.
        
        Algoritmo:
        1. Calcular Deltas Diários de T, P.
        2. Frente Fria Clássica:
           - Dia D: Pressão mínima local (cavado pré-frontal) ou começando a subir.
           - Dia D -> D+1: Temperatura cai bruscamente.
           - Dia D: Chuva significativa (frequentemente).
        """
        df_proc = df.copy()
        
        # Calcular diferenças (D - (D-1))
        df_proc['dt_temp'] = df_proc['temperatura_media'].diff()
        df_proc['dt_pressao'] = df_proc['pressao'].diff()
        
        # Shift para olhar o futuro (D+1 vs D)
        # Queremos identificar o dia D onde a frente passa.
        # Geralmente: T cai MUITO de hoje para amanhã.
        df_proc['queda_t_amnha'] = df_proc['temperatura_media'].shift(-1) - df_proc['temperatura_media']
        
        candidatos = []
        
        for i in range(1, len(df_proc)-1):
            row = df_proc.iloc[i]
            
            # Critério 1: Queda acentuada de temperatura (Pré-frontal quente -> Pós-frontal frio)
            queda_t = -row['queda_t_amnha'] # Negativo virou positivo
            
            # Critério 2: Chuva no dia da passagem
            teve_chuva = row['precipitacao'] > 5.0
            
            # Critério 3: Virada do vento (se tivessmos direção em graus, checar N->S)
            # Como temos apenas velocidade 'vento_vel', usamos intensidade como proxy de rajada
            vento_forte = row['vento_vel'] > 6.0
            
            score = 0
            if queda_t >= self.criterios['queda_temp_min']: score += 3
            if teve_chuva: score += 2
            if vento_forte: score += 1
            
            # Refinamento: Pressão sobe DEPOIS da frente
            # Dia D+1 pressão > Dia D
            delta_p_pos = df_proc.iloc[i+1]['pressao'] - row['pressao']
            if delta_p_pos >= 2.0: score += 1
            
            if score >= 4: # Limiar de detecção
                candidatos.append({
                    'data': row['data'],
                    'intensidade_queda_t': queda_t,
                    'chuva_acumulada': row['precipitacao'],
                    'delta_pressao_pos': delta_p_pos,
                    'score': score
                })
                
        df_frentes = pd.DataFrame(candidatos)
        return df_frentes

    def estatisticas_sazonais(self, df_frentes):
        """Retorna contagem de frentes por estação do ano."""
        if df_frentes.empty: return None
        
        def get_estacao(mes):
            if mes in [12, 1, 2]: return 'Verao'
            if mes in [3, 4, 5]: return 'Outono'
            if mes in [6, 7, 8]: return 'Inverno'
            return 'Primavera'
            
        df_frentes['estacao'] = df_frentes['data'].dt.month.apply(get_estacao)
        return df_frentes['estacao'].value_counts()

    def plotar_calendario_frentes(self, df_frentes, ano_foco, nome_arquivo):
        """
        Gera um gráfico estilo 'stem plot' marcando as frentes ao longo do ano.
        """
        df_ano = df_frentes[df_frentes['data'].dt.year == ano_foco]
        
        if df_ano.empty:
            print(f"Sem frentes detectadas em {ano_foco}")
            return
            
        plt.figure(figsize=(12, 4))
        
        # Datas no eixo X
        # Intensidade (queda de T) no eixo Y
        plt.stem(df_ano['data'], df_ano['intensidade_queda_t'], basefmt=" ")
        
        plt.title(f'Calendário de Passagem de Frentes Frias - {ano_foco} (Intensidade = Queda Temp)')
        plt.ylabel('Queda de Temperatura (°C)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 15)
        
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.tight_layout()
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico de frentes gerado: {caminho}")

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Detector de Frentes Frias...")
    
    # Gerar dados sintéticos de inverno com passagens de frentes
    datas = pd.date_range("2024-05-01", "2024-09-01", freq='D')
    
    # Padrão dente de serra: aquece devagar (pré-frontal), cai rápido (frente)
    temps = []
    press = []
    chuvas = []
    ventos = []
    
    t_atual = 20.0
    p_atual = 1015.0
    
    for i in range(len(datas)):
        # Ciclo de 7 dias aprox
        dia_ciclo = i % 7
        
        if dia_ciclo == 0: # Frente passa!
            t_atual -= 8.0 # Queda brusca
            p_atual += 6.0 # Pressão sobe (Alta polar entrando)
            chuva = 25.0
            vento = 10.0
        else:
            t_atual += 1.5 # Aquecimento gradual
            p_atual -= 1.0 # Pressão cai devagar
            chuva = 0.0
            vento = 3.0
            
        temps.append(t_atual + np.random.normal(0,1))
        press.append(p_atual + np.random.normal(0,1))
        chuvas.append(chuva)
        ventos.append(vento)
        
    df_teste = pd.DataFrame({
        'data': datas,
        'temperatura_media': temps,
        'pressao': press,
        'precipitacao': chuvas,
        'vento_vel': ventos
    })
    
    analisador = AnalisadorFrentesFrias()
    frentes = analisador.identificar_passagens(df_teste)
    
    print(f"\nNúmero de frentes detectadas: {len(frentes)}")
    print(frentes.head())
    
    print("\nEstatísticas por Estação:")
    print(analisador.estatisticas_sazonais(frentes))
    
    # Testar plot
    analisador.plotar_calendario_frentes(frentes, 2024, "teste_frentes_2024")
