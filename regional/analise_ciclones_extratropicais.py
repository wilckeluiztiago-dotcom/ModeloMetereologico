import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE ANÁLISE DE CICLONES EXTRATROPICAIS
============================================

O Sul do Brasil é uma das regiões com maior ciclogênese ciclogenética do hemisfério sul.
Este módulo implementa algoritmos para detectar potenciais assinaturas de ciclones
em séries temporais (quedas bruscas de pressão + ventos fortes + chuva intensa)
e calcular estatísticas de risco.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class DetectorCiclones:
    def __init__(self, limiar_pressao=1000, limiar_queda_pressao=5, limiar_vento=10):
        """
        Args:
            limiar_pressao (float): Pressão absoluta considerada baixa (hPa).
            limiar_queda_pressao (float): Queda de pressão em 24h (hPa).
            limiar_vento (float): Velocidade do vento mínima para considerar tempestade (m/s).
        """
        self.limiar_pressao = limiar_pressao
        self.limiar_queda_pressao = limiar_queda_pressao
        self.limiar_vento = limiar_vento
        self.eventos_detectados = []
        
    def detectar_eventos(self, df):
        """
        Escaneia a série temporal em busca de eventos.
        Critério composto: (Queda de Pressão OU Pressão Baixa) E (Vento Forte).
        """
        # Calcular delta pressão 24h
        df['delta_pressao'] = df['pressao'].diff(1) # Mudança diária
        
        # Identificar dias candidatos
        # Pressão caindo muito
        condicao_queda = df['delta_pressao'] <= -self.limiar_queda_pressao
        # Pressão absoluta baixa
        condicao_baixa = df['pressao'] <= self.limiar_pressao
        # Vento forte
        condicao_vento = df['vento_vel'] >= self.limiar_vento
        
        # Evento = (Queda OR Baixa) AND Vento
        mask_eventos = (condicao_queda | condicao_baixa) & condicao_vento
        
        df_eventos = df[mask_eventos].copy()
        
        # Agrupar eventos consecutivos (mesmo ciclone durando 2-3 dias)
        # Se a diferença entre datas for 1 dia, é o mesmo evento
        if df_eventos.empty:
            self.eventos_detectados = []
            return pd.DataFrame()
            
        df_eventos['grupo'] = (df_eventos['data'].diff().dt.days > 2).cumsum()
        
        # Resumir por ciclone
        resumo_ciclones = []
        for gid, grupo in df_eventos.groupby('grupo'):
            inicio = grupo['data'].min()
            fim = grupo['data'].max()
            pressao_min = grupo['pressao'].min()
            vento_max = grupo['vento_vel'].max()
            chuva_acc = grupo['precipitacao'].sum()
            duracao = (fim - inicio).days + 1
            
            # Classificação de intensidade (Simplificada Saffir-Simpson adaptada ou Beaufort)
            cat = "Ciclone Fraco"
            if vento_max > 15: cat = "Tempestade Subtropical"
            if vento_max > 25: cat = "Ciclone Intenso/Furação"
            
            resumo_ciclones.append({
                'id': gid,
                'data_inicio': inicio,
                'data_fim': fim,
                'duracao_dias': duracao,
                'pressao_minima': pressao_min,
                'vento_maximo': vento_max,
                'precipitacao_total': chuva_acc,
                'categoria': cat
            })
            
        self.eventos_detectados = pd.DataFrame(resumo_ciclones)
        return self.eventos_detectados

    def analisar_sazonalidade(self):
        """
        Analisa em quais meses os ciclones são mais frequentes.
        """
        if len(self.eventos_detectados) == 0:
            return None
            
        df = self.eventos_detectados
        df['mes'] = df['data_inicio'].dt.month
        
        contagem = df['mes'].value_counts().sort_index()
        return contagem

    def plotar_frequencia_mensal(self, nome_arquivo):
        contagem = self.analisar_sazonalidade()
        if contagem is None: return
        
        plt.figure(figsize=(10, 6))
        contagem.plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Frequência Mensal de Ciclones/Tempestades Detectadas')
        plt.xlabel('Mês')
        plt.ylabel('Número de Eventos')
        plt.xticks(range(0,12), ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'])
        plt.grid(axis='y')
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Detector de Ciclones...")
    
    # Criar 1 ano de dados
    d = pd.date_range("2023-01-01", "2023-12-31", freq='D')
    n = len(d)
    
    # Base estável
    pressao = np.random.normal(1015, 5, n)
    vento = np.random.weibull(2, n) * 5
    chuva = np.random.exponential(2, n)
    
    # Injetar um Ciclone "Bomba" em Julho
    # Dias 180, 181, 182
    idx_evento = range(180, 183)
    pressao[idx_evento] = [995, 988, 998] # Queda brusca
    vento[idx_evento] = [15, 28, 12] # Ventania (28 m/s ~ 100 km/h)
    chuva[idx_evento] = [50, 120, 30] # Muita chuva
    
    df_teste = pd.DataFrame({
        'data': d,
        'pressao': pressao,
        'vento_vel': vento,
        'precipitacao': chuva
    })
    
    detector = DetectorCiclones(limiar_pressao=1000, limiar_vento=12)
    eventos = detector.detectar_eventos(df_teste)
    
    print(f"\nEventos detectados: {len(eventos)}")
    if len(eventos) > 0:
        print(eventos.to_string())
        
        # Validar se pegou o ciclone injetado
        evento_julho = eventos[eventos['data_inicio'].dt.month == 7]
        if not evento_julho.empty:
            print("SUCESSO: Ciclone de Julho detectado corretamente.")
            print(f"Vento Máx: {evento_julho.iloc[0]['vento_maximo']:.1f} m/s")
        else:
            print("FALHA: Ciclone de Julho não detectado.")
            
    # Testar plot
    detector.plotar_frequencia_mensal("teste_ciclones_freq")
    print("Gráfico de teste gerado.")
