import multiprocessing as mp
import time
import numpy as np
import os

"""
MÓDULO DE PROCESSAMENTO PARALELO (HPC)
======================================

Gerencia a execução de tarefas pesadas distribuindo a carga entre
todos os núcleos de CPU disponíveis no supercomputador (ou workstation).
Utiliza a biblioteca multiprocessing para contornar o GIL do Python.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class OrquestradorHPC:
    def __init__(self, n_processos=None):
        self.n_processos = n_processos if n_processos else mp.cpu_count()
        print(f"[HPC] Orquestrador inicializado com {self.n_processos} núcleos.")
        
    def executar_tarefa_distribuida(self, funcao_alvo, lista_argumentos):
        """
        Executa uma função em paralelo para uma lista de inputs.
        Mapeia os dados (Scatter) e recolhe resultados (Gather).
        """
        print(f"[HPC] Iniciando Job Paralelo: {funcao_alvo.__name__}")
        t_inicio = time.time()
        
        # Pool de processos
        with mp.Pool(processes=self.n_processos) as pool:
            # Map robusto
            resultados = pool.map(funcao_alvo, lista_argumentos)
            
        t_fim = time.time()
        print(f"[HPC] Job Concluído em {t_fim - t_inicio:.4f} segundos.")
        return resultados

    def processar_chunks_matriz(self, matriz_grande, operacao_pesada):
        """
        Divide uma matriz gigante em chunks e processa em paralelo.
        Simula operação matricial de Big Data.
        """
        n_linhas = matriz_grande.shape[0]
        chunk_size = int(np.ceil(n_linhas / self.n_processos))
        
        chunks = []
        for i in range(0, n_linhas, chunk_size):
            chunks.append(matriz_grande[i:i+chunk_size])
            
        print(f"[HPC] Matriz {matriz_grande.shape} dividida em {len(chunks)} chunks.")
        
        with mp.Pool(self.n_processos) as pool:
            resultados_chunks = pool.map(operacao_pesada, chunks)
            
        # Reconstruir (Gather)
        return np.vstack(resultados_chunks)

# --- FUNÇÕES DE TESTE (TOP-LEVEL PARA MULTIPROCESSING DEVE SER PICKLABLE) ---

def _tarefa_pesada_exemplo(x):
    # Simula cálculo intenso: Fatoração ou Série de Taylor
    res = 0
    for i in range(10000):
        res += np.sin(x * i) * np.cos(x * i)
    return res

def _operacao_matriz_exemplo(sub_matriz):
    # Ex: Calcular o quadrado de cada elemento e aplicar tangete hiperbólica
    # Simula transformação não-linear em dados de satélite
    return np.tanh(sub_matriz ** 2)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Orquestrador HPC...")
    
    hpc = OrquestradorHPC()
    
    # 1. Teste Lista de Tarefas
    dados = np.linspace(0, 100, 50) # 50 tarefas
    res = hpc.executar_tarefa_distribuida(_tarefa_pesada_exemplo, dados)
    print(f"Resultados processados: {len(res)}")
    
    # 2. Teste Matriz Gigante (Simulação)
    # 1 Milhão de elementos
    print("\nAlocando matriz massiva (Simulada)...")
    matriz_teste = np.random.rand(10000, 100) 
    
    matriz_processada = hpc.processar_chunks_matriz(matriz_teste, _operacao_matriz_exemplo)
    
    print(f"Shape Saída: {matriz_processada.shape}")
    print("Processamento paralelo bem sucedido.")
