import numpy as np
import pandas as pd

def cadeia_markov_chuva(precipitacao_serie):
    """
    Calcula matriz de transição de probabilidade para estados seco/chuvoso.
    Estado 0: Seco (< 1mm)
    Estado 1: Chuvoso (>= 1mm)
    """
    estados = (precipitacao_serie >= 1.0).astype(int)
    
    # Matriz de transição
    # P00: Seco -> Seco
    # P01: Seco -> Chuvoso
    # P10: Chuvoso -> Seco
    # P11: Chuvoso -> Chuvoso
    
    transicoes = np.zeros((2, 2))
    
    for i in range(len(estados)-1):
        atual = estados[i]
        proximo = estados[i+1]
        transicoes[atual, proximo] += 1
        
    probabilidades = transicoes / transicoes.sum(axis=1, keepdims=True)
    
    return probabilidades
