# Modelo Meteorológico HPC Sul do Brasil (Fase 2)

**Autor:** Luiz Tiago Wilcke  
**Ano:** 2024  
**Licença:** MIT

---

## 🌍 Visão Geral

Este projeto consiste em um **Modelo Meteorológico de Alta Performance (HPC)** desenvolvido para simular, prever e analisar fenômenos climáticos extremos na região Sul do Brasil (RS, SC, PR). O sistema utiliza equações diferenciais parciais (Navier-Stokes), termodinâmica atmosférica e métodos avançados de Inteligência Artificial para modelagem ambiental.


1.  **Inteligência Artificial (IA/ML):** Redes Neurais, LSTMs, Autoencoders e Clustering.
2.  **Química Atmosférica:** Modelagem de ozônio, dispersão gaussiana, chuva ácida e aerossois.
3.  **Hidroenergia:** Simulação de reservatórios, vazão de rios (Muskingum) e demanda energética.
4.  **Biometeorologia:** Impacto do clima na saúde humana (UTCI/PET), vetores de doenças e agricultura.
5.  **Sensoriamento Remoto Simulado:** Algoritmos para processamento de imagens de satélite, LIDAR e Radar.

---

## 🚀 Funcionalidades Principais

### 1. Núcleo Físico e Matemático
*   **Dinâmica dos Fluidos:** Solver Navier-Stokes 2D para advecção e ventos.
*   **Termodinâmica:** Diagramas Skew-T, CAPE/CIN e índices de instabilidade.
*   **Teoria do Caos:** Atrator de Lorenz e sensibilidade às condições iniciais.
*   **Radiação:** Transferência radiativa de ondas longas e curtas (Schwarzschild).

### 2. Inteligência Artificial
*   **Predição de Séries Temporais:** Redes LSTM para chuva e temperatura.
*   **Detecção de Anomalias:** Autoencoders para identificar eventos extremos inéditos.
*   **Classificação:** Random Forest para previsão de precipitação binária.
*   **Clustering:** K-Means para zoneamento climático automático.

### 3. Química e Poluição
*   **Qualidade do Ar:** Cálculo do IQA (Índice de Qualidade do Ar) e formação de Smog.
*   **Dispersão**: Modelo de Pluma Gaussiana para fontes industriais pontuais.
*   **Fotoquímica:** Ciclo de Chapman para formação de Ozônio troposférico.

### 4. Hidrologia e Energia
*   **Balanço Hídrico:** Método de Thornthwaite-Mather e Modelo de Balde.
*   **Energia:** Estimativa de potencial hidrelétrico e curva de demanda vs. temperatura.
*   **Roteamento:** Propagação de ondas de cheia em rios (Muskingum).

### 5. Biometeorologia
*   **Conforto Térmico:** Índices UTCI e PET.
*   **Saúde:** Modelagem de risco de Dengue e excesso de mortalidade por ondas de calor.
*   **Geraçao de Risco:** Risco de incêndio florestal (Fórmula de Monte Alegre).

---

## 🛠️ Estrutura do Projeto

O projeto conta agora com mais de **60 módulos Python** organizados em domínios:

```
modeloMetereologico/
├── nucleo/                 # Core (Config, Dados, HPC)
├── fisica/                 # Navier-Stokes, Radiação, Turbulência
├── estatistica/            # GEV, Caos, Krigagem, SSA
├── regional/               # Ciclones, Frentes Frias, Agroclima
├── inteligencia_artificial/# [NOVO] Redes Neurais, LSTMs, RF
├── quimica_atmosferica/    # [NOVO] Dispersão, Ozônio, IQA
├── hidrologia/             # [NOVO] Rios, Reservatórios, Energia
├── biometeorologia/        # [NOVO] Saúde, Fogo, Conforto
├── sensoriamento_remoto/   # [NOVO] Satélite, Radar, LIDAR
└── visualizacao/           # Geradores de Gráficos e Mapas
```

---

## 📊 Visualização e Resultados

O sistema gera mais de **30 tipos de gráficos científicos**, incluindo:
*   Mapas de Temperatura e Vento (RS/SC/PR).
*   Atratores de Lorenz 3D.
*   Plumas de Dispersão de Poluentes.
*   Hidrogramas de Enchente.
*   Perfis Verticais de LIDAR.
*   Mapas de Risco Epidemiológico.

Os gráficos são salvos automaticamente na pasta `graficos_cientificos/` e `graficos_fase2/`.

---

## 💻 Como Executar

O projeto foi inteiramente desenvolvido em Python (NumPy, SciPy, Matplotlib, Pandas).

1.  **Instalar dependências:**
    ```bash
    pip install numpy matplotlib pandas scipy scikit-learn
    ```

2.  **Executar Simulação Completa:**
    ```bash
    python main.py
    ```

3.  **Gerar Gráficos da Fase 2:**
    ```bash
    python visualizacao/gerador_graficos_fase2.py
    ```

---

## 📅 Histórico de Desenvolvimento

*   **Fase 1 (Jan 2024):** Estabelecimento do núcleo físico e estatístico. 10 gráficos iniciais.
*   **Fase 2 (Expansão):** Implementação de 50 novos módulos abrangendo IA, Química e Hidrologia. Totalizando 60+ scripts e 30+ visualizações.

---

**Luiz Tiago Wilcke**  
*Desenvolvedor e Pesquisador*
