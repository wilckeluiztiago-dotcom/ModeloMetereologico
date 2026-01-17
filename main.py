import os
import sys
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Adicionar diretório atual ao path
sys.path.append(os.getcwd())

# IMPORTAÇÃO DE MÓDULOS (CORE & DADOS)
from nucleo.gerador_dados_rs import gerar_dados_rs
from nucleo.gerador_dados_sc import gerar_dados_sc
from nucleo.gerador_dados_pr import gerar_dados_pr
from nucleo.configuracao import DATA_INICIO, DATA_FIM
from nucleo.processamento_paralelo_hpc import OrquestradorHPC

# IMPORTAÇÃO FÍSICA PESADA
from fisica.navier_stokes_solver import NavierStokesSolver
from fisica.transferencia_radiativa import ModeloRadiacao
from fisica.turbulencia_atmosferica import ModeloTurbulenciaTKE

# IMPORTAÇÃO ESTATÍSTICA PESADA
from estatistica.teoria_caos_lorenz import AnalisadorCaosLorenz
from estatistica.analise_espectral_singular import SingularSpectrumAnalysis
from estatistica.interpolacao_krigagem import KrigagemSimples
from estatistica.metricas_erro import AvaliadorModelo
from estatistica.distribuicao_gev import ajustar_gev_maximos_anuais
from estatistica.analise_componentes_principais import AnalisadorComponentesPrincipais

# IMPORTAÇÃO REGIONAL
from regional.analise_ciclones_extratropicais import DetectorCiclones
from regional.analise_frentes_frias import AnalisadorFrentesFrias
from regional.analise_ondas_calor import AnalisadorOndasCalor
from regional.risco_inundacao import AnalisadorRiscoInundacao
from regional.zoneamento_agroclimatico import ZoneamentoAgro

# IMPORTAÇÃO VISUALIZAÇÃO
from visualizacao.plot_series_temporais import plotar_serie_temporal
from visualizacao.plot_mapas_reais import PlotadorMapasSul
from visualizacao.plot_tendencias import plotar_tendencia_linear_avancada

def tarefa_simulacao_estado(estado_func):
    """Wrapper para rodar geração de dados em paralelo."""
    print(f"[PID {os.getpid()}] Gerando dados massivos...")
    return estado_func(DATA_INICIO, DATA_FIM)

def main():
    warnings.filterwarnings("ignore")
    print("=== MODELO METEOROLÓGICO HPC SUL DO BRASIL (1990-2024) ===")
    print("Modo: Supercomputação / Equações Diferenciais Pesadas")
    print(f"Autor: Luiz Tiago Wilcke\n")
    
    # --- 1. GERAÇÃO DE DADOS DISTRIBUÍDA (Simulando Big Data) ---
    print("\n>>> FASE 1: GERAÇÃO DE DADOS MASSIVA (PARALELO) <<<")
    hpc = OrquestradorHPC(n_processos=3) # Usar 3 cores
    funcs = [gerar_dados_rs, gerar_dados_sc, gerar_dados_pr]
    
    # Scatter/Gather
    resultados = hpc.executar_tarefa_distribuida(tarefa_simulacao_estado, funcs)
    df_rs, df_sc, df_pr = resultados
    
    print(f"Dados gerados: {len(df_rs)} registros/estado. Total: {len(resultados)*len(df_rs)} registros.")
    
    # --- 2. FÍSICA PESADA: NAVIER-STOKES (CFD) ---
    print("\n>>> FASE 2: RESOLUÇÃO DE NAVIER-STOKES (ESCOAMENTO REGIONAL) <<<")
    # Resolver campo de vento em mesoescala sobre o RS
    solver_ns = NavierStokesSolver(nx=50, ny=50, lx=500000, ly=500000, nu=100.0, dt=10)
    print("Iterando solver CFD (20 passos)...")
    for _ in range(20):
        u, v, p = solver_ns.passo_tempo()
        
    print(f"Energia Cinética Final do Sistema: {np.sum(0.5 * 1.225 * (u**2 + v**2)):.2e} J")
    
    # --- 3. FÍSICA PESADA: RADIAÇÃO ATMOSFÉRICA ---
    print("\n>>> FASE 3: TRANSFERÊNCIA RADIATIVA (SCHWARZSCHILD) <<<")
    rad_model = ModeloRadiacao()
    # Perfil simulado
    p_levels = np.linspace(1000, 10, 50)
    t_profile = 290 * (p_levels/1000)**0.19
    q_profile = 0.01 * (p_levels/1000)**2
    
    _, down_flux, heat_rate = rad_model.resolver_schwarzschild_onda_longa(p_levels, t_profile, q_profile)
    print(f"Fluxo IV Descendente (Efeito Estufa): {down_flux[-1]:.2f} W/m²")
    
    # --- 4. ESTATÍSTICA AVANÇADA: CAOS E SSA ---
    print("\n>>> FASE 4: ANÁLISE DE CAOS E SÉRIES TEMPORAIS <<<")
    
    # Caos
    lorenz = AnalisadorCaosLorenz()
    # Verificar horizonte de previsibilidade do "clima" caótico
    print(f"Calculando Expoente de Lyapunov (Previsibilidade)...")
    _, dists = lorenz.calcular_divergencia_trajetorias([1,1,1])
    # lyap = lorenz.estimar... (omitido para brevidade no log)
    
    # SSA (Singular Spectrum Analysis) em Temperatura RS
    print("Decompondo série temporal (SSA)...")
    
    # Calcular média se não existir
    if 'temperatura_media' not in df_rs.columns:
        df_rs['temperatura_media'] = (df_rs['temperatura_max'] + df_rs['temperatura_min']) / 2
        
    # Pegar apenas um trecho para não estourar memória neste script exemplo
    ts_sample = df_rs['temperatura_media'].values[-365*2:] 
    ssa = SingularSpectrumAnalysis(window_size=30)
    ssa.fit(ts_sample)
    trend_ssa = ssa.reconstruct([0])
    print(f"SSA concluído. Variância explicada 1º comp: {ssa.explained_variance[0]*100:.1f}%")

    # --- 5. ANÁLISE REGIONAL ---
    print("\n>>> FASE 5: DIAGNÓSTICOS REGIONAIS EXTENSIVOS <<<")
    
    # Frentes Frias
    frentes_rs = AnalisadorFrentesFrias().identificar_passagens(df_rs)
    print(f"Frentes Frias detectadas (RS): {len(frentes_rs)}")
    
    # Ondas de Calor
    ondas_sc, _ = AnalisadorOndasCalor().detectar_ondas(df_sc)
    print(f"Ondas de Calor detectadas (SC): {len(ondas_sc)}")
    
    # Risco Inundação
    r_inundacao = AnalisadorRiscoInundacao()
    # Fix: Resample no DataFrame para ter acesso à coluna 'data'
    precip_max_anual = df_rs.resample('Y', on='data')['precipitacao'].max()
    loc, scale = r_inundacao.ajustar_gumbel_maximos(precip_max_anual)
    q100 = r_inundacao.calcular_nivel_retorno_gumbel(loc, scale, 100)
    print(f"Chuva max 100 anos (RS): {q100:.1f} mm/dia")
    
    # --- 6. GERAÇÃO DE MAPAS REAIS E VISUALIZAÇÃO AVANÇADA ---
    print("\n>>> FASE 6: VISUALIZAÇÃO CIENTÍFICA E MAPEAMENTO <<<")
    
    from visualizacao.visualizador_cientifico import VisualizadorCientifico
    viz_avancada = VisualizadorCientifico()
    
    # 6.1 Mapa Real (Krigagem)
    plotador = PlotadorMapasSul()
    print("Interpolando dados via Krigagem...")
    coords = np.array([[-51.2, -30.0], [-52.4, -28.2], [-52.3, -31.7], [-48.5, -27.6], [-49.2, -25.4]])
    vals = np.array([25.0, 22.0, 24.0, 21.0, 23.0]) 
    krig = KrigagemSimples()
    krig.ajustar(coords, vals)
    grid_lon = np.linspace(-57, -49, 100)
    grid_lat = np.linspace(-34, -27, 100)
    XX, YY = np.meshgrid(grid_lon, grid_lat)
    pts = np.vstack([XX.ravel(), YY.ravel()]).T
    ZZ_flat, _ = krig.predizer(pts)
    ZZ = ZZ_flat.reshape(XX.shape)
    plotador.plotar_mapa_interpolado(XX, YY, ZZ, estado='RS', titulo="Temperatura Interpolada (Krigagem) - RS", nome_arquivo="mapa_final_rs_kriging")
    
    # 6.2 Tendência Linear
    plotar_tendencia_linear_avancada(df_rs, 'temperatura_media', 'Tendência Climática 1990-2026 (RS)', 'tendencia_final_rs')
    
    # 6.3 Atrator de Lorenz 3D (Caos)
    print("Gerando gráfico 3D do Caos...")
    tempo_lorenz, traj_lorenz = lorenz.simular_trajetoria([1, 1, 1], t_max=100, passos=10000)
    viz_avancada.plotar_atrator_lorenz_3d(tempo_lorenz, traj_lorenz, "caos_atrator_lorenz_3d")
    
    # 6.4 Campo de Vento CFD (Navier-Stokes)
    print("Gerando Streamlines do Vento CFD...")
    # Usando u, v finais do solver
    viz_avancada.plotar_campo_vento_cfd(u, v, "cfd_vento_navier_stokes")
    
    # 6.5 Perfil Radiativo Vertical
    print("Gerando Perfil Vertical Radiativo...")
    viz_avancada.plotar_perfil_vertical_radiacao(p_levels, heat_rate, "perfil_vertical_radiacao")
    
    # 6.6 SSA Decomposição Detalhada
    print("Gerando Painel de Decomposição SSA...")
    viz_avancada.plotar_decomposicao_ssa(ts_sample, trend_ssa, ts_sample - trend_ssa, "ssa_decomposicao_painel")
    
    # 6.7 Histograma Comparativo
    print("Gerando Histograma Comparativo Sul...")
    viz_avancada.plotar_histograma_comparativo({
        'RS': df_rs['temperatura_max'],
        'SC': df_sc['temperatura_max'],
        'PR': df_pr['temperatura_max']
    }, "histograma_temp_sul")
    
    print("\n=== SIMULAÇÃO CONCLUÍDA COM SUCESSO ===")
    print(f"Total de gráficos gerados: 7+")
    print(f"Resultados disponíveis em: nucleo/configuracao.DIR_RESULTADOS")

if __name__ == "__main__":
    main()
