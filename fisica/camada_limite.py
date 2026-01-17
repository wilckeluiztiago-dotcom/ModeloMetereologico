import numpy as np
import pandas as pd

"""
MÓDULO DE CAMADA LIMITE PLANETÁRIA (PBL)
=========================================

Este módulo implementa equações para simular a Camada Limite Planetária (Planetary Boundary Layer - PBL).
A PBL é a parte inferior da atmosfera em contato direto com a superfície da Terra, influenciada por fricção,
aquecimento diurno e evapotranspiração.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
VERSÃO: 1.0

TEORIA
------
A altura da PBL (h) varia dinamicamente ao longo do dia.
Durante o dia, h cresce devido à convecção térmica.
Durante a noite, h colapsa para uma camada estável muito mais rasa.

Equação de Prognóstico para h (Slab Model simplificado):
dh/dt = (1 + 2A) * (HeatFlux_sfc - HeatFlux_entrainment) / (Gamma * h)

Onde:
- A: Parâmetro de arrastamento (entrainment), tipicamente 0.2
- Gamma: Gradiente de temperatura potencial na atmosfera livre
"""

class CamadaLimite:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.altura_pbl = 100.0 # Metros (inicial noturna)
        self.constante_von_karman = 0.4
        self.rugosidade = 0.1 # Metros (para grama/culturas)
        
    def calcular_altura_pbl_diurna(self, fluxo_calor_superficie, gradiente_potencial_atm_livre, dt_segundos):
        """
        Calcula a evolução da altura da camada limite (h) durante o dia (convectiva).
        
        Args:
            fluxo_calor_superficie (float): Fluxo de calor sensível cinemático (K m/s).
            gradiente_potencial_atm_livre (float): dTheta/dz acima da PBL (K/m).
            dt_segundos (float): Passo de tempo.
            
        Retorna:
            float: Nova altura da PBL.
        """
        # Evitar divisão por zero e valores físicos inválidos
        if gradiente_potencial_atm_livre < 0.001:
            gradiente_potencial_atm_livre = 0.001
            
        entrainment_ratio = 0.2
        
        # Termo de forçamento térmico
        forcing = (1 + 2 * entrainment_ratio) * fluxo_calor_superficie
        
        # Se houver resfriamento (fluxo negativo), a equação de crescimento não se aplica da mesma forma
        # PBL colapsa ou se torna estável.
        if fluxo_calor_superficie <= 0:
            return self.decaimento_noturno(dt_segundos)
            
        # Evolução: dh/dt ~ Forcing / (Gamma * h) -> h * dh = (Forcing/Gamma) * dt
        # Integrando: 0.5 * h^2_new - 0.5 * h^2_old = ...
        # Discretização explícita para simulação passo a passo
        
        dh_dt = forcing / (gradiente_potencial_atm_livre * self.altura_pbl)
        
        # Limitar taxa de crescimento para estabilidade numérica
        dh_dt = np.clip(dh_dt, -0.1, 0.5) # m/s (crescimento máximo)
        
        self.altura_pbl += dh_dt * dt_segundos
        
        # Teto físico razoável para o Sul do Brasil
        self.altura_pbl = np.clip(self.altura_pbl, 50, 3000)
        
        return self.altura_pbl

    def decaimento_noturno(self, dt):
        """
        Simula o decaimento da PBL convectiva para uma Camada Limite Estável (SBL) após o pôr do sol.
        Processo exponencial de decaimento.
        """
        altura_estavel_alvo = 150.0 # Metros
        tau = 3600.0 * 2 # Constante de tempo de relaxamento (2 horas)
        
        diff = altura_estavel_alvo - self.altura_pbl
        change = (diff / tau) * dt
        
        self.altura_pbl += change
        return self.altura_pbl
        
    def perfil_logaritmico_vento(self, u_star, z):
        """
        Calcula a velocidade do vento na altura z dentro da camada superficial,
        usando a lei logarítmica.
        
        U(z) = (u_star / k) * ln(z / z0)
        
        Args:
            u_star (float): Velocidade de fricção.
            z (float ou array): Altura(s) de interesse.
            
        Retorna:
            Velocidade do vento em m/s.
        """
        # Validar z > z0
        z_safe = np.maximum(z, self.rugosidade + 0.01)
        
        u_z = (u_star / self.constante_von_karman) * np.log(z_safe / self.rugosidade)
        return u_z

    def estimar_difusividade_turbulenta(self, z, w_star):
        """
        Perfil K (O'Brien) para difusividade turbulenta na camada mista.
        K(z) = k * w_star * z * (1 - z/h)^2
        
        Args:
            z (float): Altura.
            w_star (float): Escala de velocidade convectiva.
        """
        if z > self.altura_pbl:
            return 0.1 # Valor residual na atmosfera livre
            
        k_z = self.constante_von_karman * w_star * z * (1 - z/self.altura_pbl)**2
        return np.maximum(k_z, 0.1)

# ==============================================================================
# SEÇÃO DE TESTES E EXEMPLOS (SELF-TEST)
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Iniciando simulação de teste da Camada Limite (PBL)...")
    
    # Configuração da Simulação
    pbl_model = CamadaLimite(lat=-30.0, lon=-51.0) # Porto Alegre aprox
    
    # Parâmetros temporais (24 horas)
    dt = 600 # 10 minutos
    passos = int(24 * 3600 / dt)
    tempos = np.linspace(0, 24, passos)
    
    # Arrays para armazenar resultados
    alturas_h = []
    fluxos_calor = []
    
    # Simular ciclo diurno de fluxo de calor (positivo de dia, negativo à noite)
    # Pico ao meio dia
    fluxo_max = 0.3 # K m/s (aproximadamente 300 W/m2)
    gamma = 0.005 # 5 K / km (Atmosfera estável neutra acima)
    
    print(f"Simulando {passos} passos de tempo ({dt}s cada)...")
    
    for t_hora in tempos:
        # Fluxo solar senoidal simplificado (dia entre 6h e 18h)
        if 6 <= t_hora <= 18:
            fluxo = fluxo_max * np.sin(np.pi * (t_hora - 6) / 12)
        else:
            fluxo = -0.05 # Resfriamento noturno radiativo
            
        fluxos_calor.append(fluxo)
        
        h_atual = pbl_model.calcular_altura_pbl_diurna(fluxo, gamma, dt)
        alturas_h.append(h_atual)
        
    # Gerar Gráfico de verificação
    print("Gerando gráfico de validação interna...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Hora do Dia (Local)')
    ax1.set_ylabel('Fluxo de Calor Sensível (K m/s)', color=color)
    ax1.plot(tempos, fluxos_calor, color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Altura da PBL (m)', color=color)
    ax2.plot(tempos, alturas_h, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Simulação do Ciclo Diurno da Camada Limite Planetária (PBL)\nModelo Slab Simplificado')
    plt.tight_layout()
    # Não salvamos em arquivo no teste, apenas mostramos se fosse interativo ou rodasse em notebook
    # plt.show()
    print("Simulação concluída. Altura máxima atingida: {:.2f} m".format(max(alturas_h)))
    
    # Teste Perfil de Vento
    print("\nCalculando perfil de vento logarítmico para u* = 0.4 m/s...")
    ks = np.linspace(1, 200, 50)
    perfil_v = pbl_model.perfil_logaritmico_vento(u_star=0.4, z=ks)
    print(f"Vento a 10m: {pbl_model.perfil_logaritmico_vento(0.4, 10):.2f} m/s")
    print(f"Vento a 100m: {pbl_model.perfil_logaritmico_vento(0.4, 100):.2f} m/s")
