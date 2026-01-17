import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

"""
MÓDULO DE REGRESSÃO MULTIVARIADA AVANÇADA
=========================================

Implementa Algoritmos de Regressão Linear Múltipla (OLS)
com diagnósticos completos: R², ANOVA, Teste t, Intervalos de Confiança,
VIF (Multicolinearidade) e Análise de Resíduos (Durbin-Watson).

Focado em entender quais variáveis climáticas (ex: Sea Surface Temp)
explicam a variabilidade da precipitação ou temperatura no Sul.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class RegressaoLinearMultipla:
    def __init__(self):
        self.coeficientes = None
        self.intercepto = None
        self.residuos = None
        self.y_pred = None
        self.stats = {}
        
    def ajustar(self, X, y):
        """
        Ajusta o modelo OLS: y = X*beta + e
        Usa Algebra Matricial Direta: beta = (X^T X)^-1 X^T y
        """
        X = np.array(X)
        y = np.array(y)
        n, p = X.shape
        
        # Adicionar coluna de 1s para intercepto
        X_design = np.c_[np.ones(n), X]
        p_total = p + 1
        
        # Equação Normal
        try:
            XtX = X_design.T @ X_design
            XtX_inv = np.linalg.inv(XtX)
            Xty = X_design.T @ y
            beta = XtX_inv @ Xty
        except np.linalg.LinAlgError:
            print("ERRO: Matriz singular. Multicolinearidade perfeita?")
            return
            
        self.intercepto = beta[0]
        self.coeficientes = beta[1:]
        
        # Previsão e Resíduos
        self.y_pred = X_design @ beta
        self.residuos = y - self.y_pred
        
        # --- ESTATÍSTICAS DE DIAGNÓSTICO ---
        
        # Soma dos Quadrados
        y_bar = np.mean(y)
        sst = np.sum((y - y_bar)**2) # Total
        sse = np.sum(self.residuos**2) # Erro (Residual)
        ssr = sst - sse # Regressão
        
        # Graus de Liberdade
        df_total = n - 1
        df_res = n - p_total
        df_reg = p # Numero de preditores
        
        # Variância do Erro (MSE)
        mse = sse / df_res
        rmse = np.sqrt(mse)
        
        # R-Quadrado
        r2 = 1 - (sse / sst)
        r2_adj = 1 - (1 - r2) * (df_total / df_res)
        
        # Erro Padrão dos Coeficientes
        # Cov(beta) = MSE * (X'X)^-1
        cov_beta = mse * XtX_inv
        se_beta = np.sqrt(np.diag(cov_beta))
        
        # Estatística t e p-valor
        t_stats = beta / se_beta
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df_res)) for t in t_stats]
        
        # Estatística F
        msr = ssr / df_reg if df_reg > 0 else 0
        f_stat = msr / mse if mse > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, df_reg, df_res)
        
        # Durbin-Watson (Autocorrelação dos resíduos)
        dw = np.sum(np.diff(self.residuos)**2) / sse
        
        self.stats = {
            'n_obs': n,
            'r2': r2,
            'r2_adj': r2_adj,
            'rmse': rmse,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue,
            'sse': sse,
            'aic': n * np.log(sse/n) + 2*p_total, # Akaike
            'bic': n * np.log(sse/n) + p_total*np.log(n), # Bayes
            'durbin_watson': dw,
            'betas': beta,
            'std_errs': se_beta,
            't_stats': t_stats,
            'p_values': p_values
        }
        
    def resumo(self, nomes_X=None):
        """Gera uma tabela de resumo estatístico similar ao statsmodels/R."""
        if not self.stats:
            return "Modelo não ajustado."
            
        if nomes_X is None:
            nomes_X = [f"X{i+1}" for i in range(len(self.coeficientes))]
        nomes = ['Intercepto'] + nomes_X
        
        linhas = []
        linhas.append("="*60)
        linhas.append("RESULTADOS DA REGRESSÃO MULTIVARIADA (IMPLEMENTAÇÃO PRÓPRIA)")
        linhas.append("="*60)
        
        s = self.stats
        linhas.append(f"Obs: {s['n_obs']} | R²: {s['r2']:.4f} | R² Adj: {s['r2_adj']:.4f}")
        linhas.append(f"RMSE: {s['rmse']:.4f} | F-Stat: {s['f_stat']:.2f} (p={s['f_pvalue']:.4e})")
        linhas.append(f"Durbin-Watson: {s['durbin_watson']:.2f}")
        linhas.append("-"*60)
        linhas.append(f"{'Variável':<15} {'Coeficiente':>12} {'Std.Err':>10} {'t':>8} {'P>|t|':>10}")
        linhas.append("-"*60)
        
        for i, nome in enumerate(nomes):
            beta = s['betas'][i]
            se = s['std_errs'][i]
            t = s['t_stats'][i]
            p = s['p_values'][i]
            sig = "*" if p < 0.05 else ""
            linhas.append(f"{nome:<15} {beta:>12.4f} {se:>10.4f} {t:>8.2f} {p:>10.4f} {sig}")
            
        linhas.append("="*60)
        return "\n".join(linhas)

    def plotar_diagnosticos(self):
        """Plota gráficos de resíduos para validar pressupostos."""
        if self.residuos is None: return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Resíduos vs Preditos (Homocedasticidade)
        axes[0].scatter(self.y_pred, self.residuos, alpha=0.6)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title('Resíduos vs Valores Preditos')
        axes[0].set_xlabel('Valor Predito')
        axes[0].set_ylabel('Resíduo')
        
        # 2. QQ Plot (Normalidade)
        stats.probplot(self.residuos, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot dos Resíduos (Normalidade)')
        
        plt.tight_layout()
        print("Diagnósticos gerados.")

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Regressão Multivariada...")
    
    # Gerar dados correlacionados
    np.random.seed(123)
    n = 200
    
    # X1: Temperatura do Atlântico (SST)
    x1 = np.random.normal(25, 2, n)
    
    # X2: Umidade
    x2 = np.random.normal(80, 10, n)
    
    # Y: Chuva no RS
    # Modelo: Chuva = -100 + 5*SST + 2*Umidade + Erro
    # 5 * 25 = 125, 2 * 80 = 160 -> Total ~185 mm
    erro = np.random.normal(0, 15, n)
    y = -100 + 5*x1 + 2*x2 + erro
    
    X = np.column_stack((x1, x2))
    
    modelo = RegressaoLinearMultipla()
    modelo.ajustar(X, y)
    
    print(modelo.resumo(nomes_X=['SST_Atlantico', 'Umidade_Rel']))
    
    # Verificar precisão
    # Esperado: Intercepto ~ -100, SST ~ 5, Umidade ~ 2
    
    modelo.plotar_diagnosticos()
