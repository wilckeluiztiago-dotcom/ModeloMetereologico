import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
MÓDULO DE ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)
=================================================

Este módulo aplica a técnica de PCA (Principal Component Analysis) para reduzir a dimensionalidade
de dados meteorológicos complexos e identificar padrões dominantes (modos de variabilidade).
No contexto do clima do Sul do Brasil, pode identificar padrões como o Modo Anular Sul (SAM)
ou influências do ENSO combinadas.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class AnalisadorComponentesPrincipais:
    def __init__(self, n_componentes=0.95):
        """
        Inicializa o analisador PCA.
        
        Args:
            n_componentes (float ou int): Se float < 1, número de componentes para explicar a variância.
                                          Se int >= 1, número exato de componentes.
        """
        self.pca = PCA(n_components=n_componentes)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.dados_reduzidos = None
        
    def ajustar_transformar(self, df_dados):
        """
        Ajusta o modelo PCA aos dados e aplica a transformação.
        
        Args:
            df_dados (DataFrame): DataFrame com variáveis numéricas (Temp, Pressão, Umidade, etc.)
            
        Retorna:
            ndarray: Dados transformados no espaço dos componentes principais.
        """
        # Verificar integridade dos dados
        if df_dados.isnull().values.any():
            print("ALERTA: Dados contêm NaNs. Preenchendo com média.")
            df_dados = df_dados.fillna(df_dados.mean())
            
        self.feature_names = df_dados.columns.tolist()
        
        # 1. Normalização (Crucial para PCA em dados com unidades diferentes)
        print("Normalizando dados (Média=0, Variância=1)...")
        dados_normalizados = self.scaler.fit_transform(df_dados)
        
        # 2. Aplicação do PCA
        print(f"Aplicando PCA com n_components={self.pca.n_components}...")
        self.dados_reduzidos = self.pca.fit_transform(dados_normalizados)
        
        return self.dados_reduzidos
    
    def obter_variancia_explicada(self):
        """Retorna a razão de variância explicada por cada componente."""
        return self.pca.explained_variance_ratio_
    
    def obter_cargas(self):
        """Retorna os loadings (pesos) de cada variável original nos componentes."""
        loadings = pd.DataFrame(
            self.pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(len(self.pca.components_))],
            index=self.feature_names
        )
        return loadings

    def interpretar_resultados(self):
        """Gera um relatório textual detalhado sobre os componentes encontrados."""
        if self.pca.components_ is None:
            return "Erro: O modelo ainda não foi ajustado."
            
        relatorio = []
        relatorio.append("=== RELATÓRIO DE ANÁLISE DE COMPONENTES PRINCIPAIS ===")
        relatorio.append(f"Número de Componentes Gerados: {self.pca.n_components_}")
        relatorio.append(f"Variância Total Explicada: {sum(self.pca.explained_variance_ratio_):.4f}")
        relatorio.append("-" * 50)
        
        cargas = self.obter_cargas()
        
        for i, col in enumerate(cargas.columns):
            var_exp = self.pca.explained_variance_ratio_[i]
            relatorio.append(f"\n{col} (Explica {var_exp*100:.2f}% da variância):")
            
            # Identificar variáveis mais influentes neste componente
            influencias = cargas[col].abs().sort_values(ascending=False).head(3)
            relatorio.append("  Variáveis Dominantes:")
            for var, peso in influencias.items():
                sinal = "+" if cargas.loc[var, col] > 0 else "-"
                relatorio.append(f"    {sinal} {var}: {abs(peso):.4f}")
                
        return "\n".join(relatorio)

    def plotar_variancia_explicada(self, salvar_em=None):
        """
        Gera um gráfico Scree Plot da variância explicada.
        """
        if self.pca.explained_variance_ratio_ is None:
            print("Erro: Modelo não ajustado.")
            return
            
        plt.figure(figsize=(10, 6))
        x_ticks = range(1, len(self.pca.explained_variance_ratio_) + 1)
        var_ratio = self.pca.explained_variance_ratio_
        var_cumulativa = np.cumsum(var_ratio)
        
        plt.bar(x_ticks, var_ratio, alpha=0.6, align='center', label='Variância Individual')
        plt.step(x_ticks, var_cumulativa, where='mid', label='Variância Acumulada', color='red')
        
        plt.ylabel('Razão de Variância Explicada')
        plt.xlabel('Componentes Principais')
        plt.title('Análise PCA - Scree Plot')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if salvar_em:
            plt.savefig(salvar_em)
            print(f"Gráfico salvo em: {salvar_em}")
        else:
            print("Gráfico gerado (não salvo).")
        plt.close()

# ==============================================================================
# SELF-TEST / DEMONSTRAÇÃO
# ==============================================================================
if __name__ == "__main__":
    print("--- INICIANDO TESTE DO MÓDULO PCA ---")
    
    # 1. Gerar dados sintéticos com correlação embutida
    # Vamos simular Temperatura, Umidade, Pressão, Precipitação
    # Onde Temp e Pressão são inversamente correlacionadas, Temp e Evaporação correlacionadas
    n_amostras = 1000
    np.random.seed(42)
    
    t = np.random.normal(20, 5, n_amostras) # Temperatura
    p = 1013 - (t - 20) * 0.5 + np.random.normal(0, 2, n_amostras) # Pressão cai com calor (simplificado)
    u = 80 - (t - 15) * 1.5 + np.random.normal(0, 10, n_amostras) # Umidade cai com calor
    u = np.clip(u, 20, 100)
    chuva = np.random.exponential(5, n_amostras) * (u / 100) # Chuva depende de umidade
    
    df_teste = pd.DataFrame({
        'Temperatura': t,
        'Pressao': p,
        'Umidade': u,
        'Precipitacao': chuva,
        'Vento_u': np.random.normal(0, 5, n_amostras), # Vento aleatório (ruído)
        'Vento_v': np.random.normal(2, 5, n_amostras)
    })
    
    print("Dados originais (primeiras 5 linhas):")
    print(df_teste.head())
    
    # 2. Instanciar e Rodar PCA
    analisador = AnalisadorComponentesPrincipais(n_componentes=0.90) # Quero 90% da variância
    dados_pc = analisador.ajustar_transformar(df_teste)
    
    print(f"\nDados transformados (shape): {dados_pc.shape}")
    
    # 3. Analisar Resultados
    relatorio = analisador.interpretar_resultados()
    print("\n" + relatorio)
    
    # 4. Checar ortogonalidade (apenas didático)
    print("\nVerificando correlação entre os componentes (Deve ser ~0):")
    df_pc = pd.DataFrame(dados_pc)
    corr_pc = df_pc.corr().iloc[0, 1] if df_pc.shape[1] > 1 else 0
    print(f"Correlação PC1 x PC2: {corr_pc:.10f}")
    
    # 5. Testar gráfico
    analisador.plotar_variancia_explicada()
    
    print("\n--- TESTE FINALIZADO COM SUCESSO ---")
