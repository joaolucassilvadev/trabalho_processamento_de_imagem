# 🫀 Sistema de Reconhecimento de Batimentos Cardíacos a partir de Imagens

## Descrição do Projeto

Este projeto implementa um sistema completo de **estimativa de frequência cardíaca** utilizando **fotopletismografia (PPG)** através da câmera de um smartphone ou webcam. O sistema analisa variações sutis na cor da pele para detectar o pulso cardíaco sem necessidade de contato físico com sensores.

### 📋 Funcionalidades Principais

- **Captura de Vídeo em Tempo Real**: Utiliza OpenCV para captura de frames da câmera
- **Dois Modos de Operação**:
  - **Modo Dedo (Finger PPG)**: Dedo sobre a lente da câmera com flash ligado
  - **Modo Facial (rPPG)**: Análise remota da face do usuário
- **Pipeline de Machine Learning Completo**: Pré-processamento, treinamento e inferência
- **Modelos Implementados**: 
  - Rede Neural Convolucional 1D (CNN-1D)
  - Modelo baseado em LSTM para séries temporais
  - Versão otimizada com TensorFlow Lite para mobile
- **Interface Gráfica**: Aplicação com visualização em tempo real

## 🔬 Fundamentação Teórica

### O que é Fotopletismografia (PPG)?

A **Fotopletismografia** é uma técnica óptica não-invasiva que detecta variações no volume sanguíneo nos tecidos. O princípio básico é:

1. **Emissão de Luz**: Uma fonte de luz (LED ou luz ambiente) ilumina a pele
2. **Absorção**: A hemoglobina no sangue absorve parte dessa luz
3. **Detecção**: Um sensor (câmera) detecta a luz refletida/transmitida
4. **Variação Cíclica**: Como o volume sanguíneo varia com cada batimento cardíaco, a quantidade de luz absorvida também varia ciclicamente

### Tipos de PPG

| Tipo | Descrição | Aplicação |
|------|-----------|-----------|
| **PPG por Transmissão** | Luz atravessa o tecido | Dedo sobre câmera com flash |
| **PPG por Reflexão (rPPG)** | Luz refletida da pele | Análise facial remota |

### Sinal PPG e Frequência Cardíaca

O sinal PPG capturado contém:
- **Componente DC**: Nível médio de absorção (tecidos, sangue venoso)
- **Componente AC**: Variação pulsátil (sangue arterial) - **Este é o sinal de interesse**

A frequência cardíaca (HR) é calculada pela frequência fundamental do componente AC:

```
HR (BPM) = Frequência dominante (Hz) × 60
```

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| Python | 3.8+ | Linguagem principal |
| OpenCV | 4.x | Captura e processamento de imagens |
| TensorFlow/Keras | 2.x | Treinamento de modelos |
| TensorFlow Lite | 2.x | Inferência em dispositivos móveis |
| PyTorch | 2.x | Implementação alternativa de modelos |
| NumPy | 1.x | Processamento numérico |
| SciPy | 1.x | Processamento de sinais |
| Matplotlib | 3.x | Visualização |

## 📁 Estrutura do Projeto

```
heart_rate_project/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências do projeto
├── setup.py                     # Instalação do pacote
├── config/
│   └── config.yaml              # Configurações do sistema
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py    # Carregamento de datasets
│   │   ├── data_generator.py    # Geração de dados sintéticos
│   │   └── video_extractor.py   # Extração de frames de vídeos
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── signal_processing.py # Processamento de sinais PPG
│   │   ├── face_detector.py     # Detecção facial para rPPG
│   │   └── roi_extractor.py     # Extração de região de interesse
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py         # Modelo CNN-1D (TensorFlow)
│   │   ├── lstm_model.py        # Modelo LSTM (TensorFlow)
│   │   ├── pytorch_model.py     # Modelo PyTorch
│   │   └── tflite_converter.py  # Conversão para TFLite
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── realtime_inference.py # Inferência em tempo real
│   │   └── tflite_inference.py   # Inferência TFLite
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py     # Funções de visualização
│       └── metrics.py           # Métricas de avaliação
├── app/
│   ├── __init__.py
│   ├── main_app.py              # Aplicação principal com GUI
│   └── mobile_demo.py           # Demo para dispositivos móveis
├── data/
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados processados
│   └── synthetic/               # Dados sintéticos gerados
├── models/
│   └── saved/                   # Modelos treinados salvos
├── docs/
│   └── DOCUMENTATION.md         # Documentação completa
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # Testes unitários
└── notebooks/
    └── exploration.ipynb        # Notebooks de exploração
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Webcam ou câmera de smartphone
- (Opcional) GPU com CUDA para treinamento acelerado

### Instalação das Dependências

```bash
# Clonar ou baixar o projeto
cd heart_rate_project

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## 📊 Datasets

### Datasets Públicos Recomendados

1. **UBFC-rPPG** (Recomendado para este projeto)
   - 42 vídeos de participantes
   - 30 FPS, resolução 640x480
   - Ground truth de PPG com oxímetro
   - [Download](https://sites.google.com/view/ybenezeth/ubfcrppg)

2. **PURE Dataset**
   - 10 sujeitos, 6 atividades cada
   - Inclui movimentos de cabeça
   - [PhysioNet](https://www.physionet.org/)

3. **COHFACE**
   - 40 participantes
   - Condições controladas

### Estrutura do Dataset UBFC-rPPG

```
UBFC-rPPG/
├── subject1/
│   ├── vid.avi           # Vídeo facial
│   └── ground_truth.txt  # PPG reference (timestamps, HR, SpO2)
├── subject2/
│   └── ...
└── subjectN/
```

### Geração de Dados Sintéticos

O projeto também suporta geração de dados sintéticos para testes e aumento de dados:

```python
from src.data.data_generator import SyntheticPPGGenerator

generator = SyntheticPPGGenerator(
    heart_rate_range=(50, 120),
    noise_level=0.1
)
signals, labels = generator.generate(n_samples=1000)
```

## 💻 Uso do Sistema

### 1. Treinamento do Modelo

```bash
# Treinar com configurações padrão
python -m src.train --config config/config.yaml

# Ou com parâmetros específicos
python -m src.train \
    --data_path data/processed/ \
    --model_type cnn \
    --epochs 100 \
    --batch_size 32
```

### 2. Inferência em Tempo Real

```bash
# Iniciar aplicação com GUI
python -m app.main_app

# Ou modo linha de comando
python -m src.inference.realtime_inference --camera 0
```

### 3. Uso com Dedo sobre a Câmera

```python
from src.inference.realtime_inference import HeartRateEstimator

estimator = HeartRateEstimator(mode='finger')
estimator.start_capture()
```

### 4. Uso com Detecção Facial (rPPG)

```python
from src.inference.realtime_inference import HeartRateEstimator

estimator = HeartRateEstimator(mode='face')
estimator.start_capture()
```

## 📈 Resultados Esperados

| Métrica | Valor Típico |
|---------|--------------|
| MAE (Mean Absolute Error) | < 3 BPM |
| RMSE | < 5 BPM |
| Correlação de Pearson | > 0.95 |

## 🧪 Testes

```bash
# Executar todos os testes
python -m pytest tests/

# Testes específicos
python -m pytest tests/test_pipeline.py -v
```

## 📚 Referências

1. Verkruysse, W., Svaasand, L.O., Nelson, J.S. (2008). "Remote plethysmographic imaging using ambient light." Optics Express.

2. Poh, M.Z., McDuff, D.J., Picard, R.W. (2010). "Non-contact, automated cardiac pulse measurements using video imaging and blind source separation." Optics Express.

3. Bobbia, S., Macwan, R., Benezeth, Y., et al. (2019). "Unsupervised skin tissue segmentation for remote photoplethysmography." Pattern Recognition Letters.

## 📝 Licença

Este projeto é para fins educacionais e acadêmicos.

## 👥 Contribuição

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de submeter pull requests.

---

**Desenvolvido para fins acadêmicos** 🎓
