import seaborn as sns
import matplotlib.pyplot as plt
import os
from nucleo.configuracao import DIR_GRAFICOS

def plotar_mapa_calor_correlacao(df, nome_arquivo):
    """
    Gera heatmap de correlação entre variáveis meteorológicas.
    """
    plt.figure(figsize=(10, 8))
    corr = df[['temperatura_max', 'temperatura_min', 'precipitacao', 'umidade', 'pressao']].corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlação entre Variáveis Meteorológicas')
    
    caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
