import numpy as np

"""
MÓDULO DE INTELIGÊNCIA ARTIFICIAL: VALIDAÇÃO CRUZADA (TIME SERIES SPLIT)
========================================================================

Implementa Cross-Validation específico para séries temporais (Rolling Window).
Não embaralha os dados. Treino sempre vem antes do Teste.

Splits:
[Treino 1] -> [Teste 1]
[Treino 1 + Treino 2] -> [Teste 2]
...

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class TimeSeriesCV:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def split(self, X):
        """
        Gera índices de treino e teste.
        X: array-like de dados.
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Janela de treino cresce (Expanding Window)
            # Ou fixa (Rolling Window) - aqui faremos Expanding
            train_end = fold_size * (i + 1)
            test_end = train_end + fold_size
            
            if test_end > n_samples: break
            
            train_idx = indices[:train_end]
            test_idx = indices[train_end:test_end]
            
            yield train_idx, test_idx

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Time Series CV...")
    
    dados = np.arange(100) # 100 pontos de tempo
    tscv = TimeSeriesCV(n_splits=3)
    
    for i, (tr, te) in enumerate(tscv.split(dados)):
        print(f"Fold {i+1}:")
        print(f"  Treino: {tr[0]} a {tr[-1]} ({len(tr)} amostras)")
        print(f"  Teste:  {te[0]} a {te[-1]} ({len(te)} amostras)")
