import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import os
from nucleo.configuracao import DIR_GRAFICOS

"""
MÓDULO DE MAPEAMENTO GEOESPACIAL REAL
======================================

Gera mapas "reais" dos estados do Sul do Brasil (RS, SC, PR) renderizando
polígonos de fronteira e sobrepondo campos interpolados (Krigagem) ou
pontos de estação.

Como não podemos garantir a presença de shapefiles (.shp) ou bibliotecas
como Cartopy/Geopandas no ambiente do usuário, este módulo contém
DADOS VETORIAIS EMBUTIDOS simplificados das fronteiras estaduais.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

# DADOS VETORIAIS SIMPLIFICADOS (Latitude, Longitude)
# Polígonos aproximados para visualização "Realista" sem dependência externa
FRONTEIRAS = {
    'RS': {
        'lats': [-27.2, -27.5, -28.0, -29.2, -31.3, -32.5, -33.7, -32.0, -31.0, -30.0, -29.0, -28.5, -27.2],
        'lons': [-53.0, -54.0, -56.0, -57.6, -56.0, -53.5, -53.2, -50.5, -50.0, -49.7, -49.8, -50.5, -53.0]
    },
    'SC': {
        'lats': [-25.9, -26.5, -26.8, -27.2, -28.5, -29.0, -28.5, -28.0, -26.5, -25.9],
        'lons': [-48.7, -49.5, -51.5, -53.5, -50.5, -49.5, -48.8, -48.6, -48.5, -48.7]
    },
    'PR': {
        'lats': [-22.5, -23.0, -24.0, -25.5, -26.5, -25.9, -25.5, -24.5, -23.5, -22.5],
        'lons': [-52.5, -54.0, -54.5, -54.0, -49.5, -48.5, -48.2, -49.0, -50.0, -52.5]
    }
}

class PlotadorMapasSul:
    def __init__(self):
        pass
        
    def plotar_mapa_interpolado(self, grid_x, grid_y, grid_z, estado='SUL', titulo="Mapa Termal", unidade="°C", nome_arquivo="mapa_real"):
        """
        Plota um campo escalar (ex: Temperatura) sobre o mapa dos estados.
        
        Args:
            grid_x, grid_y: Arrays 2D de longitude e latitude.
            grid_z: Array 2D de valores.
        """
        plt.figure(figsize=(10, 10))
        
        # 1. Plotar Contornos Preenchidos (Campo Interpolado)
        # Usar levels para suavidade
        levels = np.linspace(np.min(grid_z), np.max(grid_z), 50)
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='jet', alpha=0.8)
        cbar = plt.colorbar(contour, shrink=0.7)
        cbar.set_label(unidade)
        
        # 2. Desenhar Fronteiras por cima
        self._desenhar_fronteiras(estado_foco=estado)
        
        # 3. Decoração
        plt.title(titulo.upper(), fontsize=14, fontweight='bold')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Ajustar limites
        if estado == 'RS':
            plt.xlim(-58, -49); plt.ylim(-34, -27)
        elif estado == 'SC':
            plt.xlim(-54, -48); plt.ylim(-30, -25)
        elif estado == 'PR':
            plt.xlim(-55, -48); plt.ylim(-27, -22)
        else: # Sul Inteiro
            plt.xlim(-58, -48); plt.ylim(-34, -22)
            
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Salvar
        caminho = os.path.join(DIR_GRAFICOS, f"{nome_arquivo}.png")
        plt.savefig(caminho, dpi=300)
        plt.close()
        print(f"Mapa geoespacial salvo: {caminho}")

    def _desenhar_fronteiras(self, estado_foco='SUL'):
        """Desenha os polígonos dos estados."""
        estados_para_desenhar = ['RS', 'SC', 'PR']
        
        for est in estados_para_desenhar:
            dados = FRONTEIRAS[est]
            # Fechar polígono
            lons = dados['lons'] + [dados['lons'][0]]
            lats = dados['lats'] + [dados['lats'][0]]
            
            # Estilo
            lw = 2.0 if est == estado_foco else 1.0
            color = 'black'
            
            plt.plot(lons, lats, color=color, linewidth=lw)
            
            # Label centróide aprox
            cx = np.mean(dados['lons'])
            cy = np.mean(dados['lats'])
            plt.text(cx, cy, est, fontsize=12, fontweight='bold', ha='center', color='white', 
                     path_effects=[PathEffects.withStroke(linewidth=2, foreground="black")])

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Mapeamento Realista do Sul...")
    
    # Gerar dados dummy interpolados (Grade)
    lon_range = np.linspace(-58, -48, 100)
    lat_range = np.linspace(-34, -22, 100)
    XX, YY = np.meshgrid(lon_range, lat_range)
    
    # Campo fictício: Temperatura diminui com a latitude (Sul mais frio)
    # T = 25 + 0.8 * lat (lat é negativa, então lat menor = mais negativo = mais frio)
    # -30 * 0.8 = -24 -> T=1
    # -20 * 0.8 = -16 -> T=9
    # Ajustando constante
    ZZ = 40 + 0.8 * YY + np.random.normal(0, 1, XX.shape) # Ruído
    
    plotador = PlotadorMapasSul()
    
    # Mapa Geral
    plotador.plotar_mapa_interpolado(XX, YY, ZZ, estado='SUL', 
                                     titulo="Temperatura Média Simulada (Jan 2024)", 
                                     nome_arquivo="teste_mapa_sul_real")
                                     
    print("Teste de Mapeamento concluído.")
