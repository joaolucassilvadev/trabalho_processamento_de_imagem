"""
=============================================================================
Script Principal de Treinamento
=============================================================================
Este script executa o pipeline completo de treinamento:
1. Preparação de dados (sintéticos e/ou reais)
2. Treinamento de modelos (CNN, LSTM, Híbrido)
3. Avaliação e métricas
4. Conversão para TensorFlow Lite

Uso:
    python train.py --model cnn --epochs 100
    python train.py --model lstm --epochs 100 --dataset path/to/UBFC

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_gpu():
    """Configura GPU se disponível."""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) disponível(is): {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    else:
        print("Treinamento em CPU")
        return False


def load_data(
    synthetic_samples: int = 5000,
    dataset_path: Optional[str] = None,
    signal_length: int = 300
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Carrega e prepara os dados."""
    from data.data_generator import prepare_training_data
    
    return prepare_training_data(
        synthetic_samples=synthetic_samples,
        dataset_path=dataset_path,
        signal_length=signal_length
    )


def train_cnn_model(
    data: Dict,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_dir: str = 'models/saved'
) -> Tuple:
    """Treina modelo CNN."""
    from models.cnn_model import HeartRateCNN
    
    print("\n" + "="*60)
    print("Treinando Modelo CNN")
    print("="*60)
    
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    
    # Criar modelo
    model = HeartRateCNN(
        input_length=X_train.shape[1],
        conv_filters=[32, 64, 128],
        kernel_sizes=[5, 5, 3],
        dense_units=[128, 64],
        dropout_rate=0.3,
        learning_rate=learning_rate
    )
    
    model.build()
    model.summary()
    
    # Treinar
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Avaliar
    print("\n--- Avaliação no conjunto de teste ---")
    metrics = model.evaluate(X_test, y_test)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Salvar
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'cnn_model.h5')
    model.save(model_path)
    
    return model, history, metrics


def train_lstm_model(
    data: Dict,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_dir: str = 'models/saved'
) -> Tuple:
    """Treina modelo LSTM."""
    from models.lstm_model import HeartRateLSTM
    
    print("\n" + "="*60)
    print("Treinando Modelo LSTM")
    print("="*60)
    
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    
    # Criar modelo
    model = HeartRateLSTM(
        input_length=X_train.shape[1],
        lstm_units=[64, 32],
        dense_units=[32],
        dropout_rate=0.2,
        bidirectional=True,
        learning_rate=learning_rate
    )
    
    model.build()
    model.summary()
    
    # Treinar
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Avaliar
    print("\n--- Avaliação no conjunto de teste ---")
    metrics = model.evaluate(X_test, y_test)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Salvar
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'lstm_model.h5')
    model.save(model_path)
    
    return model, history, metrics


def train_hybrid_model(
    data: Dict,
    epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = 'models/saved'
) -> Tuple:
    """Treina modelo híbrido CNN-LSTM."""
    from models.lstm_model import HeartRateCNNLSTM
    
    print("\n" + "="*60)
    print("Treinando Modelo Híbrido CNN-LSTM")
    print("="*60)
    
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    
    # Criar modelo
    model = HeartRateCNNLSTM(
        input_length=X_train.shape[1],
        conv_filters=[32, 64],
        kernel_sizes=[5, 3],
        lstm_units=[32],
        dropout_rate=0.3
    )
    
    model.build()
    model.summary()
    
    # Treinar
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Avaliar
    print("\n--- Avaliação no conjunto de teste ---")
    X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)
    predictions = model.predict(X_test)
    
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    correlation = np.corrcoef(predictions, y_test)[0, 1]
    
    metrics = {'mae': mae, 'rmse': rmse, 'pearson_r': correlation}
    
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Salvar
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'hybrid_model.h5')
    model.model.save(model_path)
    
    return model, history, metrics


def convert_to_tflite(
    model_path: str,
    output_dir: str,
    sample_data: np.ndarray
):
    """Converte modelo para TensorFlow Lite."""
    from models.tflite_converter import TFLiteConverter, optimize_for_mobile
    from tensorflow import keras
    
    print("\n" + "="*60)
    print("Convertendo para TensorFlow Lite")
    print("="*60)
    
    # Carregar modelo
    model = keras.models.load_model(model_path)
    
    # Converter com diferentes otimizações
    results = optimize_for_mobile(model, output_dir, sample_data)
    
    print("\nModelos TFLite gerados:")
    for name, info in results.items():
        print(f"  {name}: {info['size_kb']:.2f} KB")
    
    return results


def save_training_report(
    metrics: Dict,
    history: Dict,
    args: argparse.Namespace,
    output_dir: str
):
    """Salva relatório do treinamento."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_type': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'synthetic_samples': args.synthetic_samples,
        'final_metrics': {k: float(v) for k, v in metrics.items()},
        'training_history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nRelatório salvo em: {report_path}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Treinamento de modelos para estimativa de frequência cardíaca'
    )
    
    parser.add_argument('--model', '-m', type=str, default='cnn',
                       choices=['cnn', 'lstm', 'hybrid', 'all'],
                       help='Tipo de modelo a treinar')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Número de épocas de treinamento')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001,
                       help='Taxa de aprendizado')
    parser.add_argument('--synthetic-samples', '-s', type=int, default=5000,
                       help='Número de amostras sintéticas')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Caminho para dataset real (UBFC-rPPG)')
    parser.add_argument('--output', '-o', type=str, default='models/saved',
                       help='Diretório de saída')
    parser.add_argument('--convert-tflite', action='store_true',
                       help='Converter modelo para TFLite')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Sistema de Treinamento - Frequência Cardíaca via PPG")
    print("="*60)
    print(f"\nConfiguração:")
    print(f"  Modelo: {args.model}")
    print(f"  Épocas: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Amostras sintéticas: {args.synthetic_samples}")
    print(f"  Dataset: {args.dataset or 'Apenas sintético'}")
    print(f"  Saída: {args.output}")
    
    # Setup
    setup_gpu()
    
    # Carregar dados
    print("\n" + "="*60)
    print("Carregando Dados")
    print("="*60)
    
    data = load_data(
        synthetic_samples=args.synthetic_samples,
        dataset_path=args.dataset
    )
    
    # Criar diretório de saída
    os.makedirs(args.output, exist_ok=True)
    
    # Treinar modelo(s)
    if args.model == 'cnn' or args.model == 'all':
        model, history, metrics = train_cnn_model(
            data, args.epochs, args.batch_size, args.learning_rate, args.output
        )
        
        if args.convert_tflite:
            X_sample = data['train'][0][:100]
            convert_to_tflite(
                os.path.join(args.output, 'cnn_model.h5'),
                os.path.join(args.output, 'tflite_cnn'),
                X_sample
            )
        
        save_training_report(metrics, history, args, args.output)
    
    if args.model == 'lstm' or args.model == 'all':
        model, history, metrics = train_lstm_model(
            data, args.epochs, args.batch_size, args.learning_rate, args.output
        )
    
    if args.model == 'hybrid' or args.model == 'all':
        model, history, metrics = train_hybrid_model(
            data, args.epochs, args.batch_size, args.output
        )
    
    print("\n" + "="*60)
    print("Treinamento Concluído!")
    print("="*60)


if __name__ == "__main__":
    main()
