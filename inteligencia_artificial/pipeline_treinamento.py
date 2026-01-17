import numpy as np
import pandas as pd

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: PIPELINE DE TREINAMENTO AUTOMATIZADO
=======================================================================

Orquestra o fluxo completo de Machine Learning:
1. Ingestão de Dados
2. Limpeza e Engenharia de Features
3. Divisão Treino/Teste (Time-based split para séries temporais)
4. Treinamento de Modelo (RandomForest, LSTM, etc)
5. Avaliação de Métricas

Garante reprodutibilidade e escalabilidade para múltiplos estados.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class PipelineTreinamentoML:
    def __init__(self, modelo_classe, pre_processador=None):
        self.modelo = modelo_classe
        self.pre_proc = pre_processador
        self.historico_mets = {}
        
    def split_series_temporal(self, X, y, test_ratio=0.2):
        """Divisão cronológica estrita (não random!) para evitar data leakage."""
        n = len(X)
        split_idx = int(n * (1 - test_ratio))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def executar(self, df_entrada, target_col='chuva_amanha', feature_cols=['temp', 'pressao']):
        """Roda o pipeline ponta a ponta."""
        print(f"[Pipeline] Iniciando para target: {target_col}")
        
        # 1. Feature Engineering
        if self.pre_proc:
            print("[Pipeline] Processando features...")
            df = self.pre_proc.pipeline_completo(df_entrada)
        else:
            df = df_entrada.dropna()
            
        # 2. Preparar Matrizes
        # Assumindo que o target já existe ou deve ser criado (shift)
        if target_col not in df.columns:
            print("[Pipeline] Criando target (Shift -1)...")
            # Shift -1 para prever o próximo passo
            df[target_col] = df[feature_cols[0]].shift(-1) # Exemplo genérico
            df = df.dropna()
            
        X = df[feature_cols].values
        # Binarizar target para classificação se necessario
        y = (df[target_col] > 0).astype(int).values 
        
        # 3. Split
        X_train, X_test, y_train, y_test = self.split_series_temporal(X, y)
        print(f"[Pipeline] Treino: {len(X_train)} amostras. Teste: {len(X_test)} amostras.")
        
        # 4. Treino
        if hasattr(self.modelo, 'treinar'):
            self.modelo.treinar(X_train, y_train)
        elif hasattr(self.modelo, 'fit'):
            self.modelo.fit(X_train, y_train)
            
        # 5. Avaliação
        acc = 0
        if hasattr(self.modelo, 'predizer'):
            preds = self.modelo.predizer(X_test)
            acc = np.mean(preds == y_test)
            print(f"[Pipeline] Acurácia no Teste: {acc*100:.1f}%")
        
        self.historico_mets['acuracia'] = acc
        return self.modelo

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    from inteligencia_artificial.floresta_aleatoria_chuva import FlorestaAleatoriaChuva
    
    print("Testando Pipeline...")
    
    # Mock Data
    df = pd.DataFrame({
        'temp': np.random.randn(100),
        'pressao': np.random.randn(100),
        'chuva_amanha': np.random.randint(0, 2, 100)
    })
    
    rf = FlorestaAleatoriaChuva()
    pipe = PipelineTreinamentoML(rf)
    
    modelo_treinado = pipe.executar(df, target_col='chuva_amanha', feature_cols=['temp', 'pressao'])
    print("Pipeline executado com sucesso.")
