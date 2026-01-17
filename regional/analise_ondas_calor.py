import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE ANÁLISE DE ONDAS DE CALOR
===================================

Implementa critérios oficiais da OMM (Organização Meteorológica Mundial)
para identificar e classificar ondas de calor (Heat Waves).
Ex: 5 dias consecutivos com Tmax > Percentil 90 ou > Média + 5°C.

Inclui cálculo do HWDI (Heat Wave Duration Index).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AnalisadorOndasCalor:
    def __init__(self, tipo_criterio='percentil', limiar_fixo=5.0, percentil=90, duracao_minima=3):
        """
        Args:
            tipo_criterio: 'fixo' (Media + X graus) ou 'percentil' (T > P90).
            limiar_fixo: Delta T acima do normal climatológico.
            percentil: Percentil limite (ex: 90 ou 95).
            duracao_minima: Dias consecutivos para configurar onda.
        """
        self.tipo = tipo_criterio
        self.limiar_fixo = limiar_fixo
        self.percentil = percentil
        self.min_days = duracao_minima
        
    def _calcular_climatologia_diaria(self, df):
        """
        Calcula a média e percentil para cada dia do ano (1-366) baseado na série histórica.
        """
        df = df.copy()
        df['doy'] = df['data'].dt.dayofyear
        
        grouped = df.groupby('doy')['temperatura_max']
        
        climatologia = pd.DataFrame({
            'media': grouped.mean(),
            'p_limit': grouped.quantile(self.percentil / 100.0)
        })
        return climatologia
    
    def detectar_ondas(self, df):
        """
        Varre a série identificando períodos de onda de calor.
        """
        print(f"Detectando ondas de calor (Critério: {self.tipo})...")
        clim = self._calcular_climatologia_diaria(df)
        
        df_proc = df.copy()
        df_proc['doy'] = df_proc['data'].dt.dayofyear
        
        # Merge com climatologia
        df_proc = df_proc.merge(clim, left_on='doy', right_index=True, how='left')
        
        # Definir limiar do dia
        if self.tipo == 'fixo':
            df_proc['limiar'] = df_proc['media'] + self.limiar_fixo
        else:
            df_proc['limiar'] = df_proc['p_limit']
            
        # Flag dias quentes
        df_proc['quente'] = df_proc['temperatura_max'] > df_proc['limiar']
        
        # Identificar sequências
        # Técnica: agrupar IDs sequenciais
        df_proc['grupo_id'] = (df_proc['quente'] != df_proc['quente'].shift()).cumsum()
        
        ondas = []
        
        for gid, grupo in df_proc[df_proc['quente']].groupby('grupo_id'):
            if len(grupo) >= self.min_days:
                inicio = grupo['data'].min()
                fim = grupo['data'].max()
                duracao = len(grupo)
                avg_excess = (grupo['temperatura_max'] - grupo['limiar']).mean()
                max_temp = grupo['temperatura_max'].max()
                
                ondas.append({
                    'inicio': inicio,
                    'fim': fim,
                    'duracao': duracao,
                    'media_excesso': avg_excess,
                    'temp_maxima': max_temp
                })
                
        df_ondas = pd.DataFrame(ondas)
        return df_ondas, df_proc

    def plotar_evento(self, df_proc, onda_info, nome_arquivo):
        """Plota a série temporal focada em um evento específico."""
        inicio = onda_info['inicio'] - pd.Timedelta(days=5)
        fim = onda_info['fim'] + pd.Timedelta(days=5)
        
        mask = (df_proc['data'] >= inicio) & (df_proc['data'] <= fim)
        sub = df_proc[mask]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(sub['data'], sub['temperatura_max'], 'r-o', label='T Máxima')
        plt.plot(sub['data'], sub['limiar'], 'k--', label='Limiar Onda Calor')
        plt.fill_between(sub['data'], sub['temperatura_max'], sub['limiar'], 
                         where=(sub['temperatura_max'] > sub['limiar']),
                         interpolate=True, color='red', alpha=0.3)
                         
        plt.title(f"Onda de Calor: {onda_info['inicio'].date()} a {onda_info['fim'].date()} ({onda_info['duracao']} dias)")
        plt.ylabel("Temperatura (°C)")
        plt.grid(True)
        plt.legend()
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico do evento salvo: {caminho}")

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Análise de Ondas de Calor...")
    
    # Criar dados 10 anos
    datas = pd.date_range("2010-01-01", "2019-12-31", freq='D')
    # Sazonalidade
    temps = 25 + 10 * np.sin(2 * np.pi * datas.dayofyear / 365) + np.random.normal(0, 3, len(datas))
    
    # Injetar super onda de calor
    # Verão 2014 (Exemplo real histórico)
    mask_2014 = (datas >= "2014-01-15") & (datas <= "2014-02-15")
    temps[mask_2014] += 8.0 # +8 graus acima do normal por 30 dias!
    
    df_teste = pd.DataFrame({'data': datas, 'temperatura_max': temps})
    
    # Detectar
    analisador = AnalisadorOndasCalor(tipo_criterio='percentil', percentil=90, duracao_minima=5)
    ondas, df_proc = analisador.detectar_ondas(df_teste)
    
    print(f"\nOndas detectadas (Total: {len(ondas)})")
    if not ondas.empty:
        # Mostrar as top 5 mais longas
        top = ondas.sort_values('duracao', ascending=False).head(5)
        print(top[['inicio', 'duracao', 'temp_maxima']])
        
        # Plotar a maior
        maior = top.iloc[0]
        analisador.plotar_evento(df_proc, maior, "teste_onda_calor_max")
        
    print("Teste concluído.")
