import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE ZONEAMENTO AGROCLIMÁTICO
===================================

Este módulo avalia a aptidão climática de diferentes regiões do Sul do Brasil
para culturas agrícolas específicas (Soja, Trigo, Milho, Uva).
Baseia-se em critérios de temperatura, precipitação acumulada e risco de geada
durante o ciclo da cultura.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class ZoneamentoAgro:
    def __init__(self, cultura):
        """
        Inicializa o modelo para uma cultura específica.
        
        Args:
            cultura (str): 'soja', 'trigo', 'milho', 'uva'
        """
        self.cultura = cultura.lower()
        self.parametros = self._definir_parametros()
        
    def _definir_parametros(self):
        """
        Define os limites ideais para cada cultura.
        Valores aproximados para demonstração.
        """
        params = {}
        if self.cultura == 'soja':
            # Verão
            params = {
                'temp_min_ideal': 20, 'temp_max_ideal': 30,
                'chuva_ciclo_min': 450, 'chal_ciclo_max': 800,
                'meses_plantio': [10, 11, 12]
            }
        elif self.cultura == 'trigo':
            # Inverno
            params = {
                'temp_min_ideal': 10, 'temp_max_ideal': 24,
                'chuva_ciclo_min': 300, 'chal_ciclo_max': 600,
                'meses_plantio': [5, 6]
            }
        elif self.cultura == 'milho':
            params = {
                'temp_min_ideal': 18, 'temp_max_ideal': 28,
                'chuva_ciclo_min': 500, 'chal_ciclo_max': 900,
                'meses_plantio': [9, 10, 11]
            }
        # Adicione mais conforme necessário
        return params

    def avaliar_safra_anual(self, df_clima, ano):
        """
        Avalia se um ano específico foi bom, regular ou ruim para a cultura.
        Simula um ciclo de 120 dias a partir do mês médio de plantio.
        """
        if not self.parametros:
            return "Cultura não definida"
            
        mes_inicio = self.parametros['meses_plantio'][0]
        
        # Filtrar o período do ciclo (aprox 4 meses)
        # Ex: Plantio em Out (10), Colheita em Fev (2 do ano seguinte)
        # Simplificação: pegar a safra dentro do ano civil ou cruzar ano
        # Vamos pegar janela fixa start_date -> +120 dias
        
        data_plantio = pd.Timestamp(f"{ano}-{mes_inicio}-01")
        data_colheita = data_plantio + pd.Timedelta(days=120)
        
        mask = (df_clima['data'] >= data_plantio) & (df_clima['data'] <= data_colheita)
        df_ciclo = df_clima.loc[mask]
        
        if len(df_ciclo) < 100:
            return "Dados Insuficientes"
            
        # Calcular métricas acumuladas
        chuva_total = df_ciclo['precipitacao'].sum()
        temp_media = df_ciclo['temperatura_max'].mean() # Usando max como proxy de calor dia
        
        # Avaliação Lógica (Fuzzy simplificado)
        score = 0
        
        # 1. Chuva
        chuva_min = self.parametros['chuva_ciclo_min']
        chuva_max = self.parametros['chal_ciclo_max']
        
        if chuva_min <= chuva_total <= chuva_max:
            score += 2
        elif chuva_total < chuva_min * 0.7: # Seca severa
            score -= 2
        elif chuva_total > chuva_max * 1.3: # Excesso chuva
            score -= 1
        else:
            score += 0 # Regular
            
        # 2. Temperatura
        t_min_ideal = self.parametros['temp_min_ideal']
        t_max_ideal = self.parametros['temp_max_ideal']
        
        if t_min_ideal <= temp_media <= t_max_ideal:
            score += 2
        elif temp_media > t_max_ideal + 2: # Calor excessivo
            score -= 1
        
        # Classificação Final
        if score >= 3: return "Alta Aptidão"
        elif score >= 1: return "Média Aptidão"
        else: return "Baixa Aptidão (Risco)"
        
    def gerar_mapa_aptidao_historica(self, df_clima, estado):
        """
        Analisa todos os anos e gera um gráfico de barras com a qualidade das safras.
        """
        anos = df_clima['data'].dt.year.unique()
        resultados = []
        df_c = df_clima.copy()
        
        for ano in anos[:-1]: # Ignorar último ano se incompleto para ciclo
            res = self.avaliar_safra_anual(df_c, ano)
            resultados.append({'ano': ano, 'resultado': res})
            
        df_res = pd.DataFrame(resultados)
        
        # Contagem
        contagem = df_res['resultado'].value_counts()
        
        # Plot
        plt.figure(figsize=(10, 6))
        cores = {'Alta Aptidão': 'green', 'Média Aptidão': 'yellow', 'Baixa Aptidão (Risco)': 'red'}
        colors_mapped = [cores.get(x, 'gray') for x in contagem.index]
        
        contagem.plot(kind='bar', color=colors_mapped, alpha=0.8)
        plt.title(f"Aptidão Climática Histórica para {self.cultura.capitalize()} - {estado}")
        plt.ylabel("Número de Anos")
        plt.xticks(rotation=0)
        
        nome_arq = f"aptidao_{self.cultura}_{estado}"
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico de aptidão gerado: {caminho}")
        
        return df_res

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Zoneamento Agroclimático...")
    
    # Gerar dados falsos de RS
    datas = pd.date_range("2000-01-01", "2020-12-31", freq='D')
    # Chuva aleatória
    chuva = np.abs(np.random.normal(5, 10, len(datas)))
    # Temperatura sazonal
    temp = 20 + 8 * np.sin(2 * np.pi * datas.dayofyear / 365)
    
    df_teste = pd.DataFrame({'data': datas, 'precipitacao': chuva, 'temperatura_max': temp + 5})
    
    # Avaliar Soja
    modelo = ZoneamentoAgro('soja')
    
    print("\nAvaliação para o ano 2005:")
    res_2005 = modelo.avaliar_safra_anual(df_teste, 2005)
    print(f"Resultado: {res_2005}")
    
    print("\nAvaliação Histórica:")
    historico = modelo.gerar_mapa_aptidao_historica(df_teste, "RS_Simulado")
    print(historico['resultado'].value_counts())
    
    # Testar outra cultura
    modelo_trigo = ZoneamentoAgro('trigo')
    print("\nTeste Trigo 2005:")
    print(modelo_trigo.avaliar_safra_anual(df_teste, 2005))
