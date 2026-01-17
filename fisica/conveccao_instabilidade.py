import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
MÓDULO DE CONVECÇÃO E INSTABILIDADE ATMOSFÉRICA
===============================================

Calcula índices de instabilidade termodinâmica que indicam o potencial
para tempestades severas. Foca principalmente no CAPE (Convective Available Potential Energy)
e CIN (Convective Inhibition).

A atmosfera do Sul do Brasil em primavera/verão é caracterizada por alto CAPE
trazido pelo Jato de Baixos Níveis (JBN) da Amazônia, encontrando ar frio.

Fórmulas baseadas em parcel theory (teoria da parcela).

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AnalisadorInstabilidade:
    def __init__(self):
        self.g = 9.81  # Gravidade (m/s^2)
        self.cp = 1005.0 # Calor específico ar seco (J/kg K)
        self.R_d = 287.05 # Constante gás ar seco
        self.R_v = 461.5 # Constante vapor
        self.epsilon = self.R_d / self.R_v
        
    def _pressao_vapor_saturacao(self, temperatura_k):
        """Equação de Tetens/Magnus para Es."""
        temp_c = temperatura_k - 273.15
        es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        return es # hPa

    def _razao_mistura_saturacao(self, pressao_hpa, temperatura_k):
        es = self._pressao_vapor_saturacao(temperatura_k)
        ws = self.epsilon * (es / (pressao_hpa - es))
        return ws # kg/kg

    def _adiabatica_umida_gradiente(self, p, t):
        """Calcula o gradiente adiabático úmido (dT/dp)_m."""
        ws = self._razao_mistura_saturacao(p, t)
        
        numerador = 1.0 + (self.bruto_calor_latente(t) * ws) / (self.R_d * t)
        denominador = 1.0 + (self.bruto_calor_latente(t)**2 * ws) / (self.cp * self.R_v * t**2)
        
        dt_dp = (self.R_d * t / (self.cp * p)) * (numerador / denominador)
        return dt_dp

    def bruto_calor_latente(self, temp_k):
        """Calor latente de vaporização (dependente de T)."""
        temp_c = temp_k - 273.15
        val = 2.501e6 - 2370 * temp_c
        return val

    def levantar_parcela(self, p_superficie, t_superficie, td_superficie, perfil_p):
        """
        Simula a elevação de uma parcela de ar desde a superfície.
        T_parcela segue adiabática seca até NCL, depois adiabática úmida.
        """
        t_parcela = []
        
        # 1. Encontrar Nível de Condensação por Levantamento (LCL)
        # Aproximação de Espy
        ts_c = t_superficie - 273.15
        td_c = td_superficie - 273.15
        t_lcl_c = td_c - (0.0024 * (ts_c - td_c)) # Em T, não Z
        # Mas precisamos da Pressão do LCL
        # T_lcl = T_sfc - 9.8 * z_lcl (aproximadamente)
        # Melhor usar iterativo simples:
        
        curr_t = t_superficie
        curr_td = td_superficie # TD varia menos com pressão na seca (mix ratio constante)
        
        # Estado atual da parcela
        estado = "SECA"
        
        t_parcela_vals = []
        
        # Assumindo perfil_p ordenado decrescente (superficie -> topo)
        for p in perfil_p:
            if p >= p_superficie:
                # Ainda na superfície ou abaixo (ignorar)
                t_parcela_vals.append(curr_t) # Placeholder ou real T
                continue
                
            # Calcular passo de pressão
            # Nota: Implementação rigorosa exigiria integração fina step-by-step
            # Aqui vamos fazer uma aproximação baseada em intervalos do perfil fornecido
            
            # Se for o primeiro nivel acima da sup, aproximar
            prev_p = p_superficie if len(t_parcela_vals) <= 1 else perfil_p[len(t_parcela_vals)-1]
            dp = p - prev_p # Negativo
            
            if estado == "SECA":
                # Adiabática Seca: Theta constante
                # T = T0 * (P/P0)^(R/Cp)
                t_pot = curr_t * (1000/prev_p)**0.286
                new_t = t_pot * (p/1000)**0.286
                
                # Check saturação (T_parc < T_dewpoint?)
                # Dewpoint na adiabática seca cai ~2K/km, Tparcela cai ~10K/km
                # Razão de mistura w é constante
                # w = w_sfc
                w_parc = self._razao_mistura_saturacao(prev_p, curr_td) # w conservado
                ws_at_new_t = self._razao_mistura_saturacao(p, new_t)
                
                if w_parc >= ws_at_new_t:
                    estado = "UMIDA"
                    curr_t = new_t # Ponto de cruzamento
                else:
                    curr_t = new_t
            
            if estado == "UMIDA":
                # Adiabática Úmida: Integração numérica simplificada
                dt_dp = self._adiabatica_umida_gradiente(prev_p, curr_t)
                curr_t = curr_t + dt_dp * dp
                
            t_parcela_vals.append(curr_t)
            
        # Ajustar comprimento
        # Como loop acima pode pular p_superficie se não estiver na lista exata
        # Vamos fazer interpolação se necessário
        # Para simplificar este código demonstrativo, retornamos array de mesmo tamanho
        # Assumindo que perfil_p começa em p_sup
        
        return np.array(t_parcela_vals)

    def calcular_cape_cin(self, perfil_p, perfil_t_amb, perfil_t_parc):
        """
        Calcula CAPE e CIN integrando a área positiva e negativa entre T_parcela e T_ambiente no diagrama Skew-T.
        
        CAPE = Integ (g * (Tp - Te) / Te) dz
        Usamos equação hidrostática dp = -rho g dz => dz = -dp/(rho g)
        CAPE = Integ - (Tp - Te) / Te * (dp/rho) = Integ R_d * (Tp - Te) * dlnP
        
        Simplificação: CAPE = Soma [ R_d * (Tp - Te) * ln(p1/p2) ] para camadas onde Tp > Te
        """
        cape = 0.0
        cin = 0.0
        
        for i in range(len(perfil_p) - 1):
            p1 = perfil_p[i]
            p2 = perfil_p[i+1] # menor que p1
            
            # Camada média
            tp_mean = (perfil_t_parc[i] + perfil_t_parc[i+1]) / 2.0
            te_mean = (perfil_t_amb[i] + perfil_t_amb[i+1]) / 2.0
            
            # Se Tp > Te -> Empuxo positivo (CAPE)
            # Se Tp < Te -> Empuxo negativo (CIN)
            
            dif_t = tp_mean - te_mean
            
            # Log pressure thickness
            d_ln_p = np.log(p1 / p2) # Positivo pois p1 > p2
            
            # Energia da camada (J/kg)
            energia = self.R_d * dif_t * d_ln_p
            
            if dif_t > 0:
                cape += energia
            else:
                cin += abs(energia) # CIN geralmente expresso como positivo ou negativo, aqui magnitude
                
        return cape, cin

    def diagnostico_tempestade(self, cape, cin):
        """Interpreta os valores."""
        nivel_cape = "Estável"
        if cape > 500: nivel_cape = "Instabilidade Marginal"
        if cape > 1500: nivel_cape = "Instabilidade Moderada"
        if cape > 2500: nivel_cape = "Instabilidade Alta (Risco Severo)"
        
        inibicao = "Sem inibição"
        if cin > 50: inibicao = "Inibição de disparo"
        if cin > 200: inibicao = "Capping Inversion Forte (Difícil estourar)"
        
        return f"CAPE: {cape:.0f} J/kg ({nivel_cape}) | CIN: {cin:.0f} J/kg ({inibicao})"

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Análise de Instabilidade (CAPE/CIN)...")
    
    analisador = AnalisadorInstabilidade()
    
    # 1. Definir um perfil vertical sintético (tipo sondagem)
    # Superficie: 1000 hPa
    # Atmosfera Condicionalmente Instável
    
    niveis = np.linspace(1000, 200, 50) # hPa
    
    # Perfil T ambiente: Decai 6.5 K/km (Padrão)
    # T(p) aprox T0 * (p/p0)^(R*gamma/g)
    # Vamos fazer linear simples em Log-P para teste
    t_sup = 300.0 # 27C (Quente e úmido - Verão RS)
    td_sup = 295.0 # 22C (Muito úmido)
    
    # T ambiente cai rápido
    t_amb = t_sup * (niveis/1000)**(0.19) # Expoente ajustado para "frio" em altitude
    
    # Simular Parcela
    # Nota: Precisamos passar vetores alinhados
    t_parcela = analisador.levantar_parcela(1000, t_sup, td_sup, niveis)
    
    # Cálculo
    cape, cin = analisador.calcular_cape_cin(niveis, t_amb, t_parcela)
    
    print("\nResultados da Sondagem Sintética:")
    print(f"Temperatura Superfície: {t_sup-273.15:.1f}°C")
    print(f"Ponto de Orvalho Superfície: {td_sup-273.15:.1f}°C")
    print("-" * 30)
    print(analisador.diagnostico_tempestade(cape, cin))
    
    # Plotar sondagem simplificada (Diagrama Stuve simples)
    plt.figure(figsize=(6, 10))
    plt.plot(t_amb - 273.15, niveis, 'b-', label='T Ambiente')
    plt.plot(t_parcela - 273.15, niveis, 'r--', label='T Parcela (Adiabática)')
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.title(f'Diagrama Vertical Simplificado\nCAPE: {cape:.0f} J/kg')
    plt.ylabel('Pressão (hPa)')
    plt.xlabel('Temperatura (°C)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    print("Gráfico de sondagem gerado (memória).")
    
    # Teste de robustez: atmosfera estável
    print("\nTeste Atmosfera Estável (Inverno):")
    t_sup_frio = 285.0 # 12C
    t_amb_frio = t_sup_frio * (niveis/1000)**0.1 # Inversão ou isotermia leve
    t_parc_frio = analisador.levantar_parcela(1000, t_sup_frio, 275.0, niveis)
    cape_frio, cin_frio = analisador.calcular_cape_cin(niveis, t_amb_frio, t_parc_frio)
    print(analisador.diagnostico_tempestade(cape_frio, cin_frio))
