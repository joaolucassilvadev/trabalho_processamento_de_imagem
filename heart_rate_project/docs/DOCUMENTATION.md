# Documentação Completa - Sistema de Reconhecimento de Batimentos Cardíacos

## Sumário

1. [Introdução e Problema](#1-introdução-e-problema)
2. [Fundamentação Teórica](#2-fundamentação-teórica)
3. [Arquitetura do Sistema](#3-arquitetura-do-sistema)
4. [Tecnologias Utilizadas](#4-tecnologias-utilizadas)
5. [Implementação Detalhada](#5-implementação-detalhada)
6. [Datasets e Preparação de Dados](#6-datasets-e-preparação-de-dados)
7. [Modelos de Machine Learning](#7-modelos-de-machine-learning)
8. [Guia de Instalação e Execução](#8-guia-de-instalação-e-execução)
9. [Resultados e Métricas](#9-resultados-e-métricas)
10. [Referências](#10-referências)

---

## 1. Introdução e Problema

### 1.1 Descrição do Problema

O monitoramento da frequência cardíaca é fundamental para avaliação da saúde cardiovascular. Tradicionalmente, isso requer equipamentos especializados como eletrocardiógrafos (ECG) ou oxímetros de pulso. Este projeto propõe uma solução não-invasiva usando apenas a câmera de um smartphone através da técnica de Fotopletismografia (PPG).

### 1.2 Objetivos

- Desenvolver sistema de estimativa de frequência cardíaca em tempo real
- Implementar algoritmos de processamento de sinal PPG
- Treinar modelos de Deep Learning (CNN, LSTM)
- Otimizar para dispositivos móveis com TensorFlow Lite

### 1.3 Modos de Operação

| Modo | Descrição | Vantagens |
|------|-----------|-----------|
| **Finger PPG** | Dedo sobre a câmera com flash | Alta precisão |
| **Remote PPG** | Análise facial a distância | Sem contato físico |

---

## 2. Fundamentação Teórica

### 2.1 Fotopletismografia (PPG)

A Fotopletismografia é uma técnica óptica que detecta alterações no volume sanguíneo. O princípio básico:

1. Uma fonte de luz ilumina a pele
2. A hemoglobina no sangue absorve parte da luz
3. Um sensor detecta a luz refletida/transmitida
4. O volume sanguíneo varia com cada batimento cardíaco
5. Essa variação cria um sinal periódico (sinal PPG)

### 2.2 Componentes do Sinal PPG

O sinal PPG possui dois componentes principais:

**Componente DC (corrente contínua):**
- Nível médio de absorção
- Determinado por tecidos, ossos, sangue venoso
- Varia lentamente (respiração, vasomotion)

**Componente AC (corrente alternada):**
- Variação pulsátil sincronizada com o coração
- Amplitude de 1-2% do sinal total
- Este é o sinal de interesse para frequência cardíaca

### 2.3 Escolha do Canal de Cor

| Canal | Comprimento de Onda | Uso Recomendado |
|-------|---------------------|-----------------|
| Vermelho (R) | ~620-750 nm | Finger PPG (maior penetração) |
| Verde (G) | ~495-570 nm | rPPG facial (absorção hemoglobina) |
| Azul (B) | ~450-495 nm | Menor penetração, pouco usado |

### 2.4 Cálculo da Frequência Cardíaca

A frequência cardíaca (HR) é calculada identificando a frequência dominante no sinal PPG:

```
HR (BPM) = f_dominante (Hz) × 60

Onde:
- f_dominante é obtida via FFT ou detecção de picos
- Faixa válida: 0.67 Hz (40 BPM) a 3.33 Hz (200 BPM)
```

---

## 3. Arquitetura do Sistema

### 3.1 Visão Geral do Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   CAPTURA    │───►│    PRÉ-      │───►│   EXTRAÇÃO   │───►│  ESTIMATIVA  │
│   DE VÍDEO   │    │PROCESSAMENTO │    │   DE SINAL   │    │     DE HR    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                   │                    │
      ▼                    ▼                   ▼                    ▼
  - OpenCV            - Detecção         - Média RGB          - FFT
  - 30 FPS              facial           - Filtragem          - Detecção de
  - 640x480           - ROI              - Normalização         picos
                      - Skin segment.                         - Modelo ML
```

### 3.2 Módulos do Sistema

```
heart_rate_project/
├── src/
│   ├── preprocessing/
│   │   ├── signal_processing.py   # Processamento de sinais PPG
│   │   └── face_detector.py       # Detecção facial e ROI
│   ├── models/
│   │   ├── cnn_model.py           # Modelo CNN-1D
│   │   ├── lstm_model.py          # Modelo LSTM
│   │   ├── pytorch_model.py       # Implementação PyTorch
│   │   └── tflite_converter.py    # Conversão TFLite
│   ├── inference/
│   │   └── realtime_inference.py  # Inferência em tempo real
│   ├── data/
│   │   └── data_generator.py      # Geração de dados
│   └── train.py                   # Script de treinamento
├── app/
│   └── main_app.py                # Aplicação com GUI
├── config/
│   └── config.yaml                # Configurações
└── docs/
    └── DOCUMENTATION.md           # Este documento
```

---

## 4. Tecnologias Utilizadas

### 4.1 Linguagem e Ambiente

| Tecnologia | Versão | Finalidade |
|------------|--------|------------|
| Python | 3.8+ | Linguagem principal |
| pip/conda | - | Gerenciamento de pacotes |

### 4.2 Bibliotecas de Processamento de Imagem

| Biblioteca | Uso no Projeto |
|------------|----------------|
| OpenCV | Captura de vídeo, detecção facial, processamento de imagem |
| NumPy | Operações numéricas e manipulação de arrays |
| SciPy | Filtros digitais, FFT, processamento de sinais |

### 4.3 Frameworks de Deep Learning

| Framework | Uso no Projeto |
|-----------|----------------|
| TensorFlow/Keras | Treinamento de modelos CNN e LSTM |
| TensorFlow Lite | Otimização para dispositivos móveis |
| PyTorch | Implementação alternativa de modelos |

### 4.4 Interface Gráfica

| Biblioteca | Uso |
|------------|-----|
| PyQt5 | Interface gráfica desktop |
| OpenCV highgui | Visualização simples |

---

## 5. Implementação Detalhada

### 5.1 Processamento de Sinal PPG

O módulo `signal_processing.py` implementa:

**Classe PPGSignalProcessor:**

```python
# Inicialização
processor = PPGSignalProcessor(
    sampling_rate=30.0,    # Taxa de amostragem (FPS da câmera)
    low_cutoff=0.7,        # Corte inferior (42 BPM)
    high_cutoff=4.0,       # Corte superior (240 BPM)
    filter_order=4         # Ordem do filtro Butterworth
)

# Pipeline de processamento
processed_signal, heart_rate, confidence = processor.process_pipeline(raw_signal)
```

**Etapas do processamento:**

1. **Remoção de tendência (Detrend)**: Remove variações lentas de baseline
2. **Filtro passa-banda**: Isola frequências de interesse (0.7-4 Hz)
3. **Normalização**: Z-score para comparabilidade
4. **Suavização**: Média móvel para reduzir ruído

### 5.2 Detecção Facial e Extração de ROI

O módulo `face_detector.py` implementa:

**Detecção facial:**
- Haar Cascades (OpenCV clássico)
- MediaPipe (mais preciso, com landmarks)

**Regiões de interesse (ROI):**
- Testa: Melhor SNR, menos movimento
- Bochechas: Boa vascularização
- Face completa: Mais dados, mais ruído

```python
# Uso
detector = FaceDetector(method='haar')
face = detector.detect(frame)

extractor = ROIExtractor()
mask = extractor.get_roi_mask(frame, face, ROIType.FOREHEAD)
```

### 5.3 Estimativa de Frequência Cardíaca

Dois métodos implementados:

**Método FFT:**
```python
# 1. Calcular FFT do sinal
frequencies, magnitudes = processor.compute_fft(signal)

# 2. Encontrar pico na faixa válida (0.67-3.33 Hz)
peak_freq = frequencies[np.argmax(magnitudes)]

# 3. Converter para BPM
heart_rate = peak_freq * 60
```

**Método de Detecção de Picos:**
```python
# 1. Detectar picos no sinal
peaks = find_peaks(signal, distance=min_distance)

# 2. Calcular intervalos entre picos (IBI)
intervals = np.diff(peaks)

# 3. Calcular HR médio
heart_rate = (sampling_rate / np.mean(intervals)) * 60
```

---

## 6. Datasets e Preparação de Dados

### 6.1 Datasets Públicos Recomendados

**UBFC-rPPG (Recomendado):**
- 42 vídeos de participantes
- 30 FPS, 640x480
- Ground truth de PPG com oxímetro
- Download: https://sites.google.com/view/ybenezeth/ubfcrppg

**PURE Dataset:**
- 10 sujeitos, 6 atividades
- Inclui movimentos de cabeça
- Disponível no PhysioNet

### 6.2 Estrutura do Dataset UBFC-rPPG

```
UBFC-rPPG/
├── subject1/
│   ├── vid.avi           # Vídeo facial
│   └── ground_truth.txt  # Referência (timestamp, HR)
├── subject2/
│   └── ...
└── subjectN/
```

### 6.3 Geração de Dados Sintéticos

O módulo `data_generator.py` permite gerar sinais PPG sintéticos:

```python
from src.data.data_generator import SyntheticPPGGenerator

generator = SyntheticPPGGenerator(
    sampling_rate=30.0,
    signal_length=300,
    heart_rate_range=(50, 120),
    noise_level=0.1
)

# Gerar dados
signals, heart_rates = generator.generate(n_samples=5000)
```

**Características dos dados sintéticos:**
- Forma de onda PPG realista (harmônicos)
- Modulação respiratória
- Variabilidade de frequência cardíaca (HRV)
- Ruído gaussiano configurável
- Artefatos de movimento opcionais

### 6.4 Data Augmentation

Técnicas de aumento de dados implementadas:
- Variação de nível de ruído
- Deslocamento temporal
- Escala de amplitude
- Adição de artefatos

---

## 7. Modelos de Machine Learning

### 7.1 Modelo CNN-1D

**Arquitetura:**

```
Input (300, 1)
    │
    ▼
Conv1D(32, kernel=5) → BatchNorm → ReLU → MaxPool(2)
    │
    ▼
Conv1D(64, kernel=5) → BatchNorm → ReLU → MaxPool(2)
    │
    ▼
Conv1D(128, kernel=3) → BatchNorm → ReLU → MaxPool(2)
    │
    ▼
GlobalAveragePooling1D
    │
    ▼
Dense(128) → Dropout(0.3) → ReLU
    │
    ▼
Dense(64) → Dropout(0.3) → ReLU
    │
    ▼
Dense(1) → Output (Heart Rate)
```

**Uso:**
```python
from src.models.cnn_model import HeartRateCNN

model = HeartRateCNN(input_length=300)
model.build()
model.train(X_train, y_train, X_val, y_val, epochs=100)
```

### 7.2 Modelo LSTM

**Arquitetura:**
```
Input (300, 1)
    │
    ▼
Bidirectional LSTM(64) → BatchNorm
    │
    ▼
Bidirectional LSTM(32) → BatchNorm
    │
    ▼
Dense(32) → Dropout(0.2) → ReLU
    │
    ▼
Dense(1) → Output
```

### 7.3 Modelo Híbrido CNN-LSTM

Combina CNN para extração de features com LSTM para modelagem temporal:

```
Input → Conv1D blocks → LSTM → Dense → Output
```

### 7.4 Conversão para TensorFlow Lite

```python
from src.models.tflite_converter import TFLiteConverter

converter = TFLiteConverter()
converter.convert(model, 'model.tflite', quantization='dynamic')
```

**Opções de quantização:**
| Tipo | Redução de Tamanho | Perda de Precisão |
|------|-------------------|-------------------|
| none | 0% | Nenhuma |
| dynamic | ~75% | Mínima |
| float16 | ~50% | Muito baixa |
| int8 | ~75% | Baixa |

---

## 8. Guia de Instalação e Execução

### 8.1 Requisitos do Sistema

- Python 3.8 ou superior
- Webcam ou câmera de smartphone
- 4GB RAM mínimo
- (Opcional) GPU com CUDA para treinamento

### 8.2 Instalação

```bash
# 1. Clonar/baixar o projeto
cd heart_rate_project

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

### 8.3 Execução da Aplicação

**Interface gráfica (PyQt5):**
```bash
python -m app.main_app
```

**Linha de comando:**
```bash
python -m app.main_app --cli --camera 0
```

**Inferência em tempo real:**
```bash
python -m src.inference.realtime_inference --mode finger
python -m src.inference.realtime_inference --mode face
```

### 8.4 Treinamento de Modelos

```bash
# Treinar modelo CNN
python -m src.train --model cnn --epochs 100

# Treinar modelo LSTM
python -m src.train --model lstm --epochs 100

# Treinar com dataset real
python -m src.train --model cnn --dataset path/to/UBFC-rPPG

# Converter para TFLite
python -m src.train --model cnn --convert-tflite
```

### 8.5 Controles da Aplicação

| Tecla | Ação |
|-------|------|
| Q | Sair |
| M | Alternar modo (finger/face) |

---

## 9. Resultados e Métricas

### 9.1 Métricas de Avaliação

| Métrica | Descrição | Fórmula |
|---------|-----------|---------|
| MAE | Mean Absolute Error | mean(\|pred - real\|) |
| RMSE | Root Mean Square Error | sqrt(mean((pred - real)²)) |
| MAPE | Mean Absolute Percentage Error | mean(\|pred - real\| / real) × 100 |
| Pearson r | Correlação | corr(pred, real) |

### 9.2 Resultados Esperados

| Modelo | MAE (BPM) | RMSE (BPM) | Pearson r |
|--------|-----------|------------|-----------|
| CNN-1D | < 3.0 | < 5.0 | > 0.95 |
| LSTM | < 3.5 | < 5.5 | > 0.93 |
| Híbrido | < 2.5 | < 4.5 | > 0.96 |

### 9.3 Limitações

- Sensível a movimento excessivo
- Iluminação afeta qualidade do sinal
- Tom de pele pode influenciar (principalmente rPPG)
- Distância da câmera deve ser mantida

---

## 10. Referências

1. Verkruysse, W., Svaasand, L.O., Nelson, J.S. (2008). "Remote plethysmographic imaging using ambient light." Optics Express.

2. Poh, M.Z., McDuff, D.J., Picard, R.W. (2010). "Non-contact, automated cardiac pulse measurements using video imaging and blind source separation." Optics Express.

3. Bobbia, S., Macwan, R., Benezeth, Y., et al. (2019). "Unsupervised skin tissue segmentation for remote photoplethysmography." Pattern Recognition Letters.

4. Chen, W., & McDuff, D. (2018). "DeepPhys: Video-based physiological measurement using convolutional attention networks." ECCV.

5. Dataset UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg

6. Charlton, P.H., et al. (2022). "Wearable Photoplethysmography for Cardiovascular Monitoring." Proceedings of the IEEE.

---

## Anexo A: Glossário

| Termo | Definição |
|-------|-----------|
| PPG | Fotopletismografia - técnica óptica de medição |
| rPPG | Remote PPG - PPG sem contato |
| ROI | Region of Interest - região de interesse |
| BPM | Batimentos por minuto |
| FFT | Fast Fourier Transform |
| IBI | Inter-Beat Interval - intervalo entre batimentos |
| HRV | Heart Rate Variability - variabilidade da FC |

---

**Documento gerado para fins acadêmicos**
**Projeto: Reconhecimento de Batimentos Cardíacos a partir de Imagens**
