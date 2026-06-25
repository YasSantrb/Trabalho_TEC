# 🛡️ Detector de Fake News - Deep Learning (LSTM)

Este repositório contém o desenvolvimento de uma solução baseada em Inteligência Artificial para a classificação automática e detecção de notícias falsas em língua portuguesa. [cite_start]O projeto foi desenvolvido como parte dos requisitos obrigatórios da Atividade Prática da disciplina de Deep Learning[cite: 1, 104].

---

## 📖 1. Definição do Problema e Contexto
[cite_start]O avanço da desinformação digital (Fake News) gera impactos severos no comportamento social, na estabilidade econômica e nas decisões políticas[cite: 13, 16, 17]. Identificar e mitigar a propagação de boatos manualmente é uma tarefa lenta e ineficiente devido ao volume massivo de dados gerados diariamente.

* [cite_start]**Impacto e Relevância**: A desinformação afeta a sociedade civil como um todo, manipulando a opinião pública e minando a confiança em instituições de relevância real[cite: 17, 18, 19].
* [cite_start]**Justificativa do Deep Learning**: O uso de Redes Neurais Recorrentes (especificamente a arquitetura LSTM) justifica-se pela capacidade intrínseca dessas redes de capturar dependências sequenciais de longo prazo e o contexto semântico das palavras dentro de uma frase ou parágrafo, superando abordagens estatísticas tradicionais[cite: 21, 56, 57, 59].
* [cite_start]**Limitações da Solução**: O modelo baseia-se exclusivamente em padrões textuais, estilísticos e sintáticos presentes no corpo do texto[cite: 20, 44]. Ele não realiza checagem de fatos (*fact-checking*) contra bancos de dados governamentais ou agências de notícias externas.

---

## 🎯 2. Objetivo da Solução
[cite_start]Desenvolver um sistema classificador binário capaz de analisar o texto bruto de uma notícia fornecida pelo usuário, processar sua estrutura semântica e prever se ela possui características textuais de uma notícia verdadeira ou de uma desinformação (Fake News), exibindo o veredito e a probabilidade calculada em uma interface gráfica local[cite: 22, 24, 25, 29].

---

## 📊 3. Base de Dados e Pré-Processamento
[cite_start]O projeto utiliza dados reais derivados do ecossistema do jornalismo brasileiro[cite: 39, 40].
* [cite_start]**Origem dos Dados**: *Fake.br-Corpus*[cite: 42].
* [cite_start]**Estrutura e Tipo**: Dados puramente textuais de notícias rotuladas de forma binária (`fake` ou `true`)[cite: 44].
* **Pré-Processamento e Tratamento Obrigatório**: 
  * [cite_start]Conversão de todos os caracteres para minúsculo para padronização[cite: 49].
  * [cite_start]Remoção completa de pontuações, caracteres especiais e ruídos textuais[cite: 49, 52].
  * [cite_start]**Estratégia de Balanceamento e Viés**: Devido à distribuição concentrada do dataset original (onde as classes eram agrupadas em blocos sequenciais), foi aplicada uma estratégia de embaralhamento randômico completo (*shuffle*) antes da divisão, mitigando o viés de ancoragem do modelo[cite: 50, 52].
  * **Tokenização e Padding**: Criação de um vocabulário limitado a 10.000 palavras mais frequentes (`max_words`), utilizando uma palavra fora do vocabulário padrão (`<OOV>`). O truncamento e o preenchimento padronizado pós-texto (*padding post*) foram fixados em sequências de 300 tokens (`max_len`).
  * [cite_start]**Divisão dos Dados**: Os dados foram divididos na proporção de 70% para Treino, 15% para Validação e 15% para Teste (utilizando amostragem estratificada)[cite: 51].

---

## 🤖 4. Modelagem e Arquitetura em Deep Learning
[cite_start]A arquitetura da rede neural foi construída sequencialmente utilizando a API Keras/TensorFlow[cite: 53, 56, 59]:

1. **Camada de Embedding**: Dimensão de entrada de 10.000 (tamanho do vocabulário) e dimensão de saída de 128 para representação vetorial densa das palavras.
2. [cite_start]**Camada LSTM**: 64 unidades neurais para processamento e retenção da ordem sequencial do texto[cite: 59].
3. [cite_start]**Camada Dropout (0.5)**: Estratégia crucial adotada contra *overfitting*, desativando aleatoriamente 50% dos neurônios durante o treino[cite: 66].
4. [cite_start]**Camada Densa Intermediária**: 16 neurônios com função de ativação `ReLU` para introduzir não-linearidade[cite: 60].
5. [cite_start]**Camada Dropout (0.3)**: Segunda barreira de regularização contra coadaptação de pesos[cite: 66].
6. [cite_start]**Camada Densa de Saída**: 1 neurônio com função de ativação `Sigmoid`, ideal para retornar uma probabilidade contínua entre 0 e 1 em classificações binárias[cite: 60].

* **Hiperparâmetros de Treino**:
  * [cite_start]**Otimizador**: Adam[cite: 62].
  * [cite_start]**Função de Perda**: Binary Crossentropy (Entropia Cruzada Binária)[cite: 61].
  * [cite_start]**Batch Size (Tamanho do Lote)**: 64[cite: 65].
  * [cite_start]**Épocas**: Configurado para um teto de até 10 épocas[cite: 64].
  * [cite_start]**Prevenção de Overfitting**: Implementação do callback `EarlyStopping` configurado para monitorar a perda de validação (`val_loss`), com paciência de 3 épocas e restauração dos melhores pesos automáticos[cite: 66, 82]. [cite_start]O treinamento foi interrompido com sucesso na **Época 4**, garantindo a melhor capacidade de generalização do modelo[cite: 66].

---

## 📈 5. Avaliação dos Resultados
[cite_start]O modelo demonstrou um desempenho equilibrado e de alta confiabilidade ao ser submetido aos dados de teste (dados nunca vistos durante o treinamento)[cite: 67, 68, 69]:

### Relatório Quantitativo de Métricas
* [cite_start]**Acurácia Geral (Accuracy)**: 93% [cite: 74]
* **Precisão (Precision)**: 91% para Fake | [cite_start]96% para True [cite: 75]
* **Recall (Sensibilidade)**: 96% para Fake | [cite_start]90% para True [cite: 76]
* **F1-Score**: 94% para Fake | [cite_start]93% para True [cite: 77]

### [cite_start]Matriz de Confusão [cite: 78]
```text
[[488   52]   -> [Notícias Verdadeiras Corretas,  Falsos Alertas (Falsos Positivos)]
 [ 20  520]]  -> [Falsos Negativos (Passaram),   Fake News Capturadas Corretamente]
 