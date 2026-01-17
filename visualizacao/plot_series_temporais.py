import matplotlib.pyplot as plt
import os
from nucleo.configuracao import DIR_GRAFICOS

def plotar_serie_temporal(df, coluna, titulo, nome_arquivo):
    plt.figure(figsize=(12, 6))
    plt.plot(df['data'], df[coluna], label=coluna, color='blue', linewidth=0.5)
    plt.title(titulo)
    plt.xlabel('Ano')
    plt.ylabel(coluna)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"Gráfico gerado: {caminho}")

def plotar_comparacao_tres_estados(df_rs, df_sc, df_pr, coluna, titulo, nome_arquivo):
    plt.figure(figsize=(14, 7))
    # Resample anual para ficar mais limpo
    rs_anual = df_rs.set_index('data')[coluna].resample('Y').mean()
    sc_anual = df_sc.set_index('data')[coluna].resample('Y').mean()
    pr_anual = df_pr.set_index('data')[coluna].resample('Y').mean()
    
    plt.plot(rs_anual.index, rs_anual, label='RS', color='green')
    plt.plot(sc_anual.index, sc_anual, label='SC', color='red')
    plt.plot(pr_anual.index, pr_anual, label='PR', color='blue')
    
    plt.title(titulo)
    plt.ylabel(f"Média Anual de {coluna}")
    plt.legend()
    plt.grid(True)
    
    caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
