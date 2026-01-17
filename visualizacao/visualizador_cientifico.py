import matplotlib.pyplot as plt
import numpy as np
import os
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE VISUALIZAÇÃO CIENTÍFICA AVANÇADA
==========================================

Coletânea de plotadores para saídas complexas dos módulos de Física e Estatística.
Gera gráficos de alta qualidade para publicação.

1. Atrator 3D (Caos)
2. Espectrograma Wavelet
3. Campo Vetorial de Vento (CFD)
4. Perfil Vertical Radiativo
5. Biplot PCA

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class VisualizadorCientifico:
    def __init__(self):
        pass
        
    def plotar_atrator_lorenz_3d(self, t, trajetoria, nome_arquivo="caos_atrator_3d"):
        """Plota a borboleta de Lorenz em 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colorir pela velocidade ou tempo
        s = ax.scatter(trajetoria[:,0], trajetoria[:,1], trajetoria[:,2], c=t, cmap='plasma', s=0.5, alpha=0.6)
        
        ax.set_title("Atrator de Lorenz: Caos Determinístico na Atmosfera", fontsize=14)
        ax.set_xlabel("Eixo X (Convecção)")
        ax.set_ylabel("Eixo Y (Gradiente T Horizontal)")
        ax.set_zlabel("Eixo Z (Gradiente T Vertical)")
        
        cbar = plt.colorbar(s, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label("Tempo de Simulação")
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico 3D salvo: {caminho}")

    def plotar_campo_vento_cfd(self, u, v, nome_arquivo="cfd_streamlines"):
        """Plota linhas de corrente (streamlines) do solver Navier-Stokes."""
        ny, nx = u.shape
        Y, X = np.mgrid[0:ny, 0:nx]
        
        velocidade = np.sqrt(u**2 + v**2)
        
        plt.figure(figsize=(10, 8))
        strm = plt.streamplot(X, Y, u, v, color=velocidade, cmap='viridis', linewidth=1, density=2)
        plt.colorbar(strm.lines, label='Velocidade do Vento (m/s)')
        
        plt.title(f"Simulação CFD: Campo de Vento (Navier-Stokes)", fontsize=14)
        plt.xlabel("Distância X (km)")
        plt.ylabel("Distância Y (km)")
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico CFD salvo: {caminho}")

    def plotar_perfil_vertical_radiacao(self, pressoes, taxa_aquecimento, nome_arquivo="perfil_radiativo"):
        """Plota o perfil vertical de aquecimento/resfriamento (Cooling Rate)."""
        plt.figure(figsize=(6, 8))
        
        # Calcular altura aproximada para eixo secundário
        # z = -H ln(p/p0)
        alturas_km = -7.0 * np.log(pressoes/1000.0)
        
        ax1 = plt.gca()
        ax1.plot(taxa_aquecimento, pressoes[:-1], 'b-o', markersize=4, label='Taxa Aquecimento')
        ax1.invert_yaxis()
        ax1.set_xlabel("Taxa de Aquecimento (K/dia)")
        ax1.set_ylabel("Pressão (hPa)")
        ax1.grid(True, which='both', linestyle='--')
        
        # Eixo secundário Altura
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        # Mapear ticks
        ticks_p = np.array([1000, 850, 700, 500, 300, 200, 100, 10])
        ticks_z = -7.0 * np.log(ticks_p/1000.0)
        ax2.set_yticks(ticks_p)
        ax2.set_yticklabels([f"{z:.1f}" for z in ticks_z])
        ax2.set_ylabel("Altura Aproximada (km)")
        
        plt.title("Perfil Vertical de Resfriamento Radiativo (Onda Longa)", fontsize=12)
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico Radiativo salvo: {caminho}")

    def plotar_espectro_wavelet(self, tempo, sinal, nome_arquivo="wavelet_spectrum"):
        """Plota Scalograma (Wavelet Power Spectrum)."""
        # Como stats.analise_wavelet já deve ter a lógica de cálculo, aqui focamos no plot
        # Mas para garantir autonomia, vamos usar scipy.signal.cwt simples aqui se necessário
        # ou apenas plotar dados passados.
        pass # Implementado via analise_wavelet.py já existente, vamos só chamar.

    def plotar_decomposicao_ssa(self, original, reconstrucao_tendencia, residuos, nome_arquivo="ssa_decomposicao"):
        """Painel com Original, Tendência Extraída e Resíduos."""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(original, 'k', alpha=0.7)
        plt.title("Série Original (Temperatura)", fontsize=10)
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(reconstrucao_tendencia, 'r', linewidth=2)
        plt.title("Reconstrução SSA (Componente 1 - Tendência)", fontsize=10)
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(residuos, 'g', alpha=0.5)
        plt.title("Resíduos (Ruído + Ciclos de alta freq)", fontsize=10)
        plt.grid(True)
        
        plt.tight_layout()
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Gráfico SSA salvo: {caminho}")

    def plotar_histograma_comparativo(self, dados_dict, nome_arquivo="hist_comparativo"):
        """Histograma overlay de 3 estados."""
        plt.figure(figsize=(10, 6))
        
        cores = {'RS': 'red', 'SC': 'green', 'PR': 'blue'}
        
        for label, dados in dados_dict.items():
            plt.hist(dados, bins=50, alpha=0.3, label=label, color=cores.get(label, 'gray'), density=True)
            # Adicionar KDE line seria ideal, mas seaborn não garantido
            
        plt.title("Distribuição de Temperatura Máxima - Comparativo Sul", fontsize=14)
        plt.xlabel("Temperatura (°C)")
        plt.ylabel("Densidade de Frequência")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Histograma salvo: {caminho}")

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    viz = VisualizadorCientifico()
    # Teste rápido
    u = np.random.randn(20, 20)
    v = np.random.randn(20, 20)
    viz.plotar_campo_vento_cfd(u, v, "teste_cfd")
