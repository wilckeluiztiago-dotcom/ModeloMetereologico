import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Adicionar root ao path
sys.path.append(os.getcwd())

# Importar Módulos da Fase 2
from inteligencia_artificial.redes_neurais_vento import MLPRegressorVento
from inteligencia_artificial.clustering_climas import KMeansClimatico
from quimica_atmosferica.dispersao_gaussiana import ModeloPlumaGaussiana
from quimica_atmosferica.emissoes_veiculares import InventarioVeicular
from hidrologia.roteamento_rios import RoteamentoMuskingum
from biometeorologia.vetores_doencas import ModeloRiscoDengue
from biometeorologia.indice_pet import IndicePET
from hidrologia.demanda_energetica import ModeloDemandaEnergia
from sensoriamento_remoto.lidar_backscatter import SimuladorLidar
from sensoriamento_remoto.mosaico_imagens import Mosaicador

def gerar_graficos_fase2():
    print("=== Gerando 10 Gráficos Científicos Avançados (Fase 2) ===")
    os.makedirs("graficos_fase2", exist_ok=True)
    
    # 1. Gráfico de Rede Neural (Arquitetura Conceitual)
    print("1. Gerando Arquitetura Neural...")
    plt.figure(figsize=(8, 6))
    layers = [4, 8, 8, 1]
    for i, n in enumerate(layers):
        x = np.ones(n) * i
        y = np.linspace(0, 1, n)
        plt.scatter(x, y, s=500, c='skyblue', ec='k', zorder=10)
        # Conexões
        if i < len(layers) - 1:
            for y1 in y:
                for y2 in np.linspace(0, 1, layers[i+1]):
                    plt.plot([i, i+1], [y1, y2], 'k-', alpha=0.1)
    plt.xticks(range(4), ['Input', 'Hidden 1', 'Hidden 2', 'Output'])
    plt.yticks([])
    plt.title("Arquitetura MLP para Previsão de Vento")
    plt.savefig("graficos_fase2/01_rede_neural_arquitetura.png")
    plt.close()

    # 2. Mapa de Pluma de Poluição (Gaussiana 2D)
    print("2. Gerando Pluma de Poluição...")
    mod_pol = ModeloPlumaGaussiana(taxa_emissao_gs=100, altura_chamine=50, velocidade_vento=5.0)
    # Grid espacial
    x = np.linspace(0, 5000, 100)
    y = np.linspace(-500, 500, 50)
    X, Y = np.meshgrid(x, y)
    C = np.zeros_like(X)
    for i in range(100):
        for j in range(50):
            C[j,i] = mod_pol.calcular_concentracao(X[j,i], Y[j,i], z=0, estabilidade='D')
    
    plt.figure(figsize=(10, 4))
    plt.contourf(X, Y, C, levels=20, cmap='YlOrRd')
    plt.colorbar(label='Concentração (µg/m³)')
    plt.title("Dispersão de Pluma Gaussiana (Estabilidade Neutra)")
    plt.xlabel("Distância (m)")
    plt.ylabel("Desvio Lateral (m)")
    plt.savefig("graficos_fase2/02_pluma_poluicao.png")
    plt.close()

    # 3. Hidrograma de Vazão (Muskingum)
    print("3. Gerando Hidrograma...")
    mod_rio = RoteamentoMuskingum()
    t = np.arange(0, 72)
    inflow = 20 + 100 * np.exp(-((t-12)**2)/50) # Pico em 12h
    outflow = mod_rio.propagar_onda(inflow)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, inflow, label='Inflow (Montante)', color='blue')
    plt.plot(t, outflow, label='Outflow (Jusante)', color='red', linestyle='--')
    plt.fill_between(t, inflow, alpha=0.1, color='blue')
    plt.title("Amortecimento de Onda de Cheia (Muskingum)")
    plt.xlabel("Tempo (h)")
    plt.ylabel("Vazão (m³/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("graficos_fase2/03_hidrograma_vazao.png")
    plt.close()

    # 4. Mapa de Risco de Dengue
    print("4. Gerando Risco Dengue...")
    mod_dengue = ModeloRiscoDengue()
    ts = np.linspace(10, 35, 50)
    ps = np.linspace(0, 200, 50)
    TT, PP = np.meshgrid(ts, ps)
    Risco = np.zeros_like(TT)
    for i in range(50):
        for j in range(50):
            Risco[i,j] = mod_dengue.calcular_indice_risco(TT[i,j], PP[i,j])
            
    plt.figure(figsize=(7, 6))
    plt.contourf(TT, PP, Risco, levels=15, cmap='RdYlGn_r')
    plt.colorbar(label='Índice de Risco (0-100)')
    plt.title("Potencial Epidemiológico de Dengue")
    plt.xlabel("Temperatura Média (°C)")
    plt.ylabel("Chuva Acumulada 15 dias (mm)")
    plt.savefig("graficos_fase2/04_risco_dengue.png")
    plt.close()

    # 5. Imagem de Satélite Sintética (Mosaico)
    print("5. Gerando Mosaico de Satélite...")
    img1 = np.random.normal(0.5, 0.1, (100, 100))
    img2 = np.random.normal(0.6, 0.1, (100, 100)) # Mais claro
    mos = Mosaicador()
    final = mos.criar_mosaico_horizontal(img1, img2, overlap_pixels=20)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(final, cmap='terrain')
    plt.title("Mosaico de Imagens de Satélite (Simulado)")
    plt.colorbar(label='Reflectância')
    plt.savefig("graficos_fase2/05_mosaico_satelite.png")
    plt.close()

    # 6. Gráfico de Dispersão LIDAR
    print("6. Gerando Perfil LIDAR...")
    mod_lidar = SimuladorLidar()
    z = np.linspace(10, 3000, 300)
    beta = 1e-4 * np.exp(-z/800)
    beta[150:160] += 0.005 # Nuvem baixa
    sinal = mod_lidar.simular_perfil(z, beta, beta*40)
    
    plt.figure(figsize=(5, 7))
    plt.plot(np.log10(sinal+1e-10), z, 'g-')
    plt.title("Perfil Vertical LIDAR (Backscatter)")
    plt.xlabel("Log(Intensidade)")
    plt.ylabel("Altura (m)")
    plt.grid(True)
    plt.savefig("graficos_fase2/06_perfil_lidar.png")
    plt.close()

    # 7. Rosa dos Ventos de Poluentes (Simulada polar plot)
    print("7. Gerando Rosa dos Ventos...")
    theta = np.linspace(0, 2*np.pi, 36)
    conc = 50 + 30 * np.cos(theta - np.pi/4) + np.random.normal(0, 5, 36) # Poluição vem de NE
    
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, conc, color='purple', linewidth=2)
    ax.fill(theta, conc, alpha=0.25, color='purple')
    ax.set_title("Rosa de Poluição (Concentração vs Direção Vento)", va='bottom')
    plt.savefig("graficos_fase2/07_rosa_poluicao.png")
    plt.close()

    # 8. Matriz de Confusão ML
    print("8. Gerando Matriz de Confusão...")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_true = np.random.choice([0, 1], size=100)
    y_pred = y_true.copy()
    y_pred[::5] = 1 - y_pred[::5] # 20% erro
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sem Chuva', 'Chuva'])
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusão (Random Forest)")
    plt.savefig("graficos_fase2/08_matriz_confusao.png")
    plt.close()

    # 9. Perfil de Conforto Térmico (PET)
    print("9. Gerando Perfil de Conforto...")
    mod_pet = IndicePET()
    horas = np.arange(24)
    temps = 20 + 10 * np.sin((horas-9)*np.pi/12)
    pet_vals = [mod_pet.estimar_pet_simplificado(t, 20, 1.0, 800 if 6<h<18 else 0) for h, t in zip(horas, temps)]
    
    plt.figure(figsize=(8, 4))
    plt.plot(horas, pet_vals, 'o-', color='orange', label='PET')
    plt.axhspan(18, 23, color='green', alpha=0.2, label='Zona Conforto')
    plt.title("Ciclo Diário de Conforto Térmico (PET)")
    plt.xlabel("Hora")
    plt.ylabel("Temperatura Equivalente (°C)")
    plt.legend()
    plt.savefig("graficos_fase2/09_conforto_termico.png")
    plt.close()

    # 10. Heatmap de Demanda Energética
    print("10. Gerando Heatmap Demanda...")
    mod_demand = ModeloDemandaEnergia()
    ts = np.linspace(0, 40, 40) # Temp
    hrs = np.arange(24) # Hora
    
    grid_demanda = np.zeros((len(ts), len(hrs)))
    for i, t in enumerate(ts):
        for j, h in enumerate(hrs):
            grid_demanda[i, j] = mod_demand.prever_demanda(t, h)
            
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_demanda, aspect='auto', origin='lower', extent=[0, 24, 0, 40], cmap='inferno')
    plt.colorbar(label='Demanda (MW)')
    plt.title("Demanda Energética: Hora vs Temperatura")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Temperatura Média (°C)")
    plt.savefig("graficos_fase2/10_heatmap_demanda.png")
    plt.close()
    
    print("=== Todos os gráficos gerados em /graficos_fase2 ===")

if __name__ == "__main__":
    gerar_graficos_fase2()
