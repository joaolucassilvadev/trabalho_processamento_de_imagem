"""
=============================================================================
Testes Unitários do Sistema de Frequência Cardíaca
=============================================================================
Execute com: python -m pytest tests/test_pipeline.py -v

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
import sys
import os

# Adicionar path do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


class TestSyntheticDataGenerator:
    """Testes para o gerador de dados sintéticos."""
    
    def test_single_signal_generation(self):
        """Testa geração de um único sinal."""
        from data.data_generator import SyntheticPPGGenerator
        
        generator = SyntheticPPGGenerator(
            sampling_rate=30,
            signal_length=300,
            heart_rate_range=(60, 100)
        )
        
        signal, hr = generator.generate_single(heart_rate=75)
        
        assert signal.shape == (300,), "Shape do sinal incorreto"
        assert 50 <= hr <= 120, "HR fora da faixa esperada"
        assert not np.isnan(signal).any(), "Sinal contém NaN"
    
    def test_batch_generation(self):
        """Testa geração de múltiplos sinais."""
        from data.data_generator import SyntheticPPGGenerator
        
        generator = SyntheticPPGGenerator()
        signals, hrs = generator.generate(n_samples=100)
        
        assert signals.shape == (100, 300), "Shape dos sinais incorreto"
        assert hrs.shape == (100,), "Shape dos labels incorreto"
        assert len(np.unique(hrs)) > 10, "Pouca variação nos labels"
    
    def test_augmentation(self):
        """Testa data augmentation."""
        from data.data_generator import SyntheticPPGGenerator
        
        generator = SyntheticPPGGenerator()
        signals, hrs = generator.generate_with_augmentation(
            n_base_samples=50,
            augmentation_factor=4
        )
        
        expected_size = 50 * 4
        assert len(signals) == expected_size, f"Esperado {expected_size}, obteve {len(signals)}"


class TestSignalProcessing:
    """Testes para processamento de sinais."""
    
    def test_bandpass_filter(self):
        """Testa filtro passa-banda."""
        from preprocessing.signal_processing import PPGSignalProcessor
        
        processor = PPGSignalProcessor(sampling_rate=30)
        
        # Criar sinal de teste
        t = np.arange(300) / 30
        signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz = 72 BPM
        
        filtered = processor.bandpass_filter(signal)
        
        assert len(filtered) == len(signal), "Tamanho alterado após filtragem"
        assert not np.isnan(filtered).any(), "Filtro produziu NaN"
    
    def test_heart_rate_estimation_fft(self):
        """Testa estimativa de HR via FFT."""
        from preprocessing.signal_processing import PPGSignalProcessor
        
        processor = PPGSignalProcessor(sampling_rate=30)
        
        # Criar sinal com frequência conhecida (72 BPM = 1.2 Hz)
        t = np.arange(300) / 30
        signal = np.sin(2 * np.pi * 1.2 * t)
        
        hr, confidence = processor.estimate_heart_rate_fft(signal)
        
        assert 60 <= hr <= 84, f"HR estimado {hr} muito diferente de 72 BPM"
    
    def test_pipeline(self):
        """Testa pipeline completo."""
        from preprocessing.signal_processing import PPGSignalProcessor, create_synthetic_ppg
        
        processor = PPGSignalProcessor(sampling_rate=30)
        
        # Criar sinal sintético
        signal = create_synthetic_ppg(
            duration=10,
            sampling_rate=30,
            heart_rate=80,
            noise_level=0.1
        )
        
        processed, hr, confidence = processor.process_pipeline(signal)
        
        assert processed.shape == signal.shape, "Shape alterado no pipeline"
        assert 60 <= hr <= 100, f"HR {hr} fora da faixa esperada"
        assert 0 <= confidence <= 1, "Confiança fora de [0, 1]"


class TestCNNModel:
    """Testes para o modelo CNN."""
    
    def test_model_build(self):
        """Testa construção do modelo."""
        from models.cnn_model import HeartRateCNN
        
        model = HeartRateCNN(input_length=300)
        keras_model = model.build()
        
        assert keras_model is not None, "Modelo não foi construído"
        assert len(keras_model.layers) > 5, "Modelo muito simples"
    
    def test_model_prediction(self):
        """Testa predição do modelo."""
        from models.cnn_model import HeartRateCNN
        
        model = HeartRateCNN(input_length=300)
        model.build()
        
        # Dados de teste
        X = np.random.randn(10, 300)
        predictions = model.predict(X)
        
        assert predictions.shape == (10,), f"Shape incorreto: {predictions.shape}"
        assert not np.isnan(predictions).any(), "Predições contêm NaN"
    
    def test_model_lite(self):
        """Testa modelo leve."""
        from models.cnn_model import HeartRateCNNLite
        
        model = HeartRateCNNLite(input_length=300)
        keras_model = model.build()
        
        # Contar parâmetros
        n_params = keras_model.count_params()
        assert n_params < 100000, f"Modelo lite muito grande: {n_params} params"


class TestLSTMModel:
    """Testes para o modelo LSTM."""
    
    def test_lstm_build(self):
        """Testa construção do LSTM."""
        from models.lstm_model import HeartRateLSTM
        
        model = HeartRateLSTM(input_length=300)
        keras_model = model.build()
        
        assert keras_model is not None
    
    def test_lstm_prediction(self):
        """Testa predição do LSTM."""
        from models.lstm_model import HeartRateLSTM
        
        model = HeartRateLSTM(input_length=300)
        model.build()
        
        X = np.random.randn(5, 300)
        predictions = model.predict(X)
        
        assert predictions.shape == (5,)
    
    def test_hybrid_model(self):
        """Testa modelo híbrido CNN-LSTM."""
        from models.lstm_model import HeartRateCNNLSTM
        
        model = HeartRateCNNLSTM(input_length=300)
        keras_model = model.build()
        
        assert keras_model is not None


class TestTFLiteConversion:
    """Testes para conversão TFLite."""
    
    def test_conversion(self):
        """Testa conversão básica."""
        import tensorflow as tf
        from tensorflow import keras
        from models.tflite_converter import TFLiteConverter
        import tempfile
        
        # Criar modelo simples
        model = keras.Sequential([
            keras.layers.Input(shape=(300, 1)),
            keras.layers.Conv1D(16, 5, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Converter
        converter = TFLiteConverter()
        
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
            output_path = f.name
        
        converter.convert(model, output_path, quantization='dynamic')
        
        # Verificar arquivo
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Limpar
        os.remove(output_path)


class TestPyTorchModels:
    """Testes para modelos PyTorch."""
    
    def test_cnn_forward(self):
        """Testa forward pass do CNN PyTorch."""
        import torch
        from models.pytorch_model import HeartRateCNN1D
        
        model = HeartRateCNN1D(input_length=300)
        
        # Dados de teste
        X = torch.randn(5, 1, 300)
        output = model(X)
        
        assert output.shape == (5,), f"Shape incorreto: {output.shape}"
    
    def test_lstm_forward(self):
        """Testa forward pass do LSTM PyTorch."""
        import torch
        from models.pytorch_model import HeartRateLSTM
        
        model = HeartRateLSTM(input_length=300)
        
        X = torch.randn(5, 1, 300)
        output = model(X)
        
        assert output.shape == (5,)
    
    def test_dataset(self):
        """Testa dataset PyTorch."""
        import torch
        from models.pytorch_model import PPGDataset
        
        signals = np.random.randn(100, 300).astype(np.float32)
        labels = np.random.uniform(60, 100, 100).astype(np.float32)
        
        dataset = PPGDataset(signals, labels)
        
        assert len(dataset) == 100
        
        signal, label = dataset[0]
        assert signal.shape == (1, 300)


class TestQualityAssessment:
    """Testes para avaliação de qualidade."""
    
    def test_quality_score(self):
        """Testa cálculo de score de qualidade."""
        from preprocessing.signal_processing import PPGQualityAssessor, create_synthetic_ppg
        
        assessor = PPGQualityAssessor(sampling_rate=30)
        
        # Sinal de boa qualidade
        good_signal = create_synthetic_ppg(10, 30, 75, noise_level=0.05)
        quality, metrics = assessor.assess_signal_quality(good_signal)
        
        assert 0 <= quality <= 1, "Score fora de [0, 1]"
        assert 'snr' in metrics
        assert 'periodicity' in metrics
    
    def test_noisy_signal_detection(self):
        """Testa detecção de sinal ruidoso."""
        from preprocessing.signal_processing import PPGQualityAssessor
        
        assessor = PPGQualityAssessor(sampling_rate=30)
        
        # Sinal muito ruidoso
        noisy_signal = np.random.randn(300)
        quality, _ = assessor.assess_signal_quality(noisy_signal)
        
        assert quality < 0.5, "Sinal ruidoso deveria ter baixa qualidade"


class TestIntegration:
    """Testes de integração."""
    
    def test_full_pipeline(self):
        """Testa pipeline completo de dados até predição."""
        from data.data_generator import SyntheticPPGGenerator
        from models.cnn_model import HeartRateCNN
        
        # Gerar dados
        generator = SyntheticPPGGenerator()
        X, y = generator.generate(n_samples=50)
        
        # Dividir
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        
        # Criar e treinar modelo (poucas épocas para teste)
        model = HeartRateCNN(input_length=300)
        model.build()
        
        # Treinar brevemente
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=2,
            batch_size=8,
            verbose=0
        )
        
        assert 'loss' in history
        
        # Predizer
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)


def run_quick_tests():
    """Executa testes rápidos sem pytest."""
    print("="*60)
    print("Executando Testes Rápidos")
    print("="*60)
    
    tests = [
        ("Geração de dados sintéticos", test_synthetic_data),
        ("Processamento de sinal", test_signal_processing),
        ("Modelo CNN", test_cnn_model),
        ("Modelo LSTM", test_lstm_model),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print("="*60)
    print(f"Resultado: {passed} passaram, {failed} falharam")
    print("="*60)
    
    return failed == 0


def test_synthetic_data():
    from data.data_generator import SyntheticPPGGenerator
    gen = SyntheticPPGGenerator()
    signals, hrs = gen.generate(n_samples=10)
    assert signals.shape == (10, 300)


def test_signal_processing():
    from preprocessing.signal_processing import PPGSignalProcessor
    proc = PPGSignalProcessor(sampling_rate=30)
    t = np.arange(300) / 30
    signal = np.sin(2 * np.pi * 1.2 * t)
    filtered = proc.bandpass_filter(signal)
    assert len(filtered) == 300


def test_cnn_model():
    from models.cnn_model import HeartRateCNN
    model = HeartRateCNN(input_length=300)
    model.build()
    X = np.random.randn(5, 300)
    pred = model.predict(X)
    assert pred.shape == (5,)


def test_lstm_model():
    from models.lstm_model import HeartRateLSTM
    model = HeartRateLSTM(input_length=300)
    model.build()
    X = np.random.randn(5, 300)
    pred = model.predict(X)
    assert pred.shape == (5,)


if __name__ == "__main__":
    # Executar testes rápidos
    success = run_quick_tests()
    
    if not success:
        sys.exit(1)
