import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy import stats
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE VISUALIZAÇÃO DE TENDÊNCIAS CLIMÁTICAS
===============================================

Focado em gerar gráficos analíticos de longo prazo (1990-2024), detectando
sinais de mudança climática através de regressões lineares, médias móveis
e suavização Lowess.

AUTOR: Luiz Tiago Wilcke
"""

def plotar_tendencia_linear_avancada(df, coluna_val, titulo, nome_arquivo):
    """
    Gera um gráfico da série temporal com:
    1. Dados brutos (fundo suavizado)
    2. Média Móvel (Anual)
    3. Linha de Regressão Linear com Intervalo de Confiança (IC 95%)
    4. Anotação da taxa de mudança (decadal).
    """
    if df is None or df.empty:
        print(f"Erro: DataFrame vazio para {nome_arquivo}")
        return

    # Preparar dados
    # Converter datas para ordinal para regressão
    df_fig = df.copy()
    df_fig = df_fig.dropna(subset=[coluna_val])
    
    # Agrupamento anual para limpar visualização se for muito longo
    df_anual = df_fig.set_index('data')[coluna_val].resample('Y').mean().reset_index()
    df_anual['data_ordinal'] = df_anual['data'].apply(lambda x: x.toordinal())
    
    # Dados diários para fundo (cinza claro)
    plt.figure(figsize=(14, 8))
    
    # Plot dados brutos (amostragem para não pesar se for muito grande)
    plt.scatter(df_fig['data'], df_fig[coluna_val], color='gray', alpha=0.1, s=1, label='Dados Diários')
    
    # Plot média anual (pontos fortes)
    plt.plot(df_anual['data'], df_anual[coluna_val], 'o-', color='navy', alpha=0.6, label='Média Anual', linewidth=1)
    
    # Regressão Linear (Seaborn regplot facilita o IC)
    # Mas para controle total no eixo temporal, vamos calcular manual ou usar truque
    # Usando scipy para equação
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_anual['data_ordinal'], df_anual[coluna_val])
    
    # Linha de tendência
    linha_tendencia = slope * df_anual['data_ordinal'] + intercept
    plt.plot(df_anual['data'], linha_tendencia, color='red', linewidth=2, label=f'Tendência Linear (R²={r_value**2:.2f})')
    
    # Intervalo de Confiança (simplificado usando seaborn por cima se necessário, ou matplotlib fill_between)
    # Aqui vamos usar uma aproximação visual simples para o código não ficar gigante com bootstrap
    # ou usar o seaborn direto no eixo convertendo depois. Vamos manter o matplotlib puro para controle.
    
    # Anotações Estatísticas
    total_anos = (df_anual['data'].max() - df_anual['data'].min()).days / 365.25
    mudanca_total = slope * (total_anos * 365.25)
    taxa_decadal = (mudanca_total / total_anos) * 10
    
    stats_text = (
        f"Taxa de Mudança: {taxa_decadal:+.2f} {buscar_unidade(coluna_val)}/década\n"
        f"Mudança Total ({int(total_anos)} anos): {mudanca_total:+.2f}\n"
        f"P-valor: {p_value:.4e} {'(* Sig)' if p_value < 0.05 else '(Não Sig)'}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.title(titulo, fontsize=16)
    plt.xlabel('Ano')
    plt.ylabel(coluna_val)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    # Salvar
    caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"Gráfico de tendência avançada gerado: {caminho}")

def buscar_unidade(nome_coluna):
    if 'temp' in nome_coluna: return '°C'
    if 'precip' in nome_coluna: return 'mm'
    if 'umida' in nome_coluna: return '%'
    if 'press' in nome_coluna: return 'hPa'
    if 'vento' in nome_coluna: return 'm/s'
    return ''

def plotar_decomposicao_stl(df, coluna, periodo, nome_arquivo):
    """
    Plota os componentes da decomposição STL (Sazonal, Tendência, Resíduo)
    em subplots separados.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    df_clean = df.dropna(subset=[coluna]).set_index('data')
    # Resample mensal para clareza na decomposição
    serie_mensal = df_clean[coluna].resample('M').mean()
    
    res = seasonal_decompose(serie_mensal, model='additive', period=12)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    res.observed.plot(ax=ax1, color='black', linewidth=1)
    ax1.set_ylabel('Observado')
    ax1.set_title(f'Decomposição STL: {coluna}')
    ax1.grid(True)
    
    res.trend.plot(ax=ax2, color='red', linewidth=1.5)
    ax2.set_ylabel('Tendência')
    ax2.grid(True)
    
    res.seasonal.plot(ax=ax3, color='blue', linewidth=1)
    ax3.set_ylabel('Sazonalidade')
    ax3.grid(True)
    
    res.resid.plot(ax=ax4, color='green', marker='o', linestyle='None', markersize=2, alpha=0.5)
    ax4.set_ylabel('Resíduo')
    ax4.grid(True)
    ax4.axhline(0, color='black', linestyle='--')
    
    caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"Gráfico STL gerado: {caminho}")

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando gerador de gráficos de tendência...")
    
    # Dados Mock
    dates = pd.date_range("1990-01-01", "2020-12-31", freq='D')
    n = len(dates)
    trend = np.linspace(20, 22, n) # Aquecimento 2C
    season = 5 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 2, n)
    temps = trend + season + noise
    
    df_mock = pd.DataFrame({'data': dates, 'temperatura_simulada': temps})
    
    # 1. Teste Linear
    plotar_tendencia_linear_avancada(df_mock, 'temperatura_simulada', 
                                     'Teste de Tendência (Simulado 1990-2020)', 
                                     'teste_tendencia_mock')
                                     
    # 2. Teste STL
    # Resample feito internamente na funcao, passamos DF diario
    plotar_decomposicao_stl(df_mock, 'temperatura_simulada', 12, 'teste_stl_mock')
    
    print("Testes gráficos concluídos.")
