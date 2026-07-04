# Classifica-o_de_eventos_acusticos_projeto_SCC5986


Repositório oficial do projeto correspondente ao artigo **"Análise de impacto de redução de dimensionalidade e balanceamento de dados em dados acústicos"**, desenvolvido no Instituto de Ciências Matemáticas e de Computação da Universidade de São Paulo (ICMC - USP) para a disciplina SCC5986 (Ciência de dados e aprendizado de máquina).

## Resumo

A classificação de eventos acústicos desempenha um papel fundamental em aplicações modernas, como segurança pública, monitoramento ambiental e gestão de cidades inteligentes. No entanto, o desenvolvimento de modelos robustos enfrenta desafios intrínsecos de representação dos dados e forte desbalanceamento entre classes. 

Este projeto investiga o impacto de:
- **Representações de dados:** Features acústicas brutas vs. Embeddings de Vision Transformers (ViT) e Espectrogramas Mel 2D.
- **Técnicas de balanceamento:** SMOTE, ADASYN e funções de perda modificadas (Focal Loss, Class-Balanced Loss).
- **Complexidade de classificadores:** Modelos tradicionais (SVM, MLP, XGBoost, Random Forest, etc.) refinados via Optuna vs. Redes Neurais Convolucionais (CNNs) e Vision Transformers.

## Estrutura do Repositório

Com base na arquitetura do projeto e nos experimentos abordados, os diretórios estão organizados da seguinte maneira:

- `Exp1/`: Scripts e resultados do **Experimento 1**.
- `Exp2/`: Scripts e resultados do **Experimento 2**.
- `Exp3/`: Scripts e resultados do **Experimento 3** (CNNs e Funções de Perda)
- `Exp4/`: Scripts e resultados do **Experimento 4** (Fine-Tuning ViT).
- `Estudos_Salvos/`: Notebooks adicionais, arquivos de experimentação do optuna e modelos gerados ao longo das análises.
- `dados/`: Dataset depois da extração de features e pré-processamento e Script de pré-processamento.

## Base de Dados

O projeto utiliza a base de dados **DataSEC** ([disponível no Zenodo](https://zenodo.org/records/15340689)). 
- **Tamanho:** 5.024 amostras de áudio (.wav, 44,1 kHz).
- **Classes:** 40 categorias distintas (variando de ambientes urbanos a rurais).


## Recursos 

- **Linguagem:** Python
- **Machine Learning:** scikit-learn, XGBoost
- **Deep Learning:** PyTorch / TensorFlow (Arquiteturas CNN e ViT)
- **Otimização:** Optuna (Busca de hiperparâmetros), GriDSearch
- **Processamento de Áudio:** AudioTools

## 👥 Autores

- **Gustavo L. Lopes** - gustavo.lima.lopes@usp.br
- **Ana Julia G. Tendulini** - anajulia.gonzalezt@usp.br
