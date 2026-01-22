"""
=============================================================================
Conversor para TensorFlow Lite
=============================================================================
Este módulo implementa a conversão de modelos Keras/TensorFlow para
TensorFlow Lite, permitindo inferência em dispositivos móveis.

TensorFlow Lite é otimizado para:
- Baixa latência
- Pequeno tamanho de modelo
- Inferência eficiente em CPU/GPU mobile

Técnicas de otimização:
- Quantização dinâmica
- Quantização float16
- Quantização int8 (requer dataset representativo)

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Callable, List
import os


class TFLiteConverter:
    """
    Conversor de modelos Keras para TensorFlow Lite.
    
    Exemplo de Uso:
        >>> converter = TFLiteConverter()
        >>> converter.convert(model, 'model.tflite', quantization='dynamic')
    """
    
    def __init__(self):
        """Inicializa o conversor."""
        pass
    
    def convert(
        self,
        model: keras.Model,
        output_path: str,
        quantization: str = 'dynamic',
        representative_dataset: Optional[Callable] = None
    ) -> str:
        """
        Converte modelo Keras para TensorFlow Lite.
        
        Args:
            model: Modelo Keras treinado
            output_path: Caminho para salvar o modelo .tflite
            quantization: Tipo de quantização:
                - 'none': Sem quantização (float32)
                - 'dynamic': Quantização dinâmica (recomendado)
                - 'float16': Quantização float16
                - 'int8': Quantização inteira (requer dataset)
            representative_dataset: Função geradora de dados para int8
        
        Returns:
            Caminho do modelo salvo
        """
        # Criar conversor
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configurar quantização
        if quantization == 'none':
            # Sem otimização
            pass
            
        elif quantization == 'dynamic':
            # Quantização dinâmica de pesos
            # Reduz tamanho ~4x com perda mínima de precisão
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif quantization == 'float16':
            # Quantização float16
            # Reduz tamanho ~2x, mantém boa precisão
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization == 'int8':
            # Quantização inteira completa
            # Máxima redução de tamanho, requer calibração
            if representative_dataset is None:
                raise ValueError(
                    "Quantização int8 requer representative_dataset"
                )
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Converter
        tflite_model = converter.convert()
        
        # Salvar
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Informações do modelo
        model_size = os.path.getsize(output_path) / 1024  # KB
        print(f"Modelo TFLite salvo: {output_path}")
        print(f"Tamanho: {model_size:.2f} KB")
        print(f"Quantização: {quantization}")
        
        return output_path
    
    def create_representative_dataset(
        self,
        X_sample: np.ndarray,
        n_samples: int = 100
    ) -> Callable:
        """
        Cria função geradora de dataset representativo para quantização int8.
        
        Args:
            X_sample: Amostra dos dados de entrada
            n_samples: Número de amostras a usar
        
        Returns:
            Função geradora
        """
        # Preparar amostras
        if len(X_sample) > n_samples:
            indices = np.random.choice(len(X_sample), n_samples, replace=False)
            X_sample = X_sample[indices]
        
        # Garantir shape correto
        if len(X_sample.shape) == 2:
            X_sample = X_sample.reshape(-1, X_sample.shape[1], 1)
        
        def representative_dataset():
            for sample in X_sample:
                yield [sample[np.newaxis, :, :].astype(np.float32)]
        
        return representative_dataset
    
    def verify_conversion(
        self,
        keras_model: keras.Model,
        tflite_path: str,
        test_input: np.ndarray,
        tolerance: float = 0.01
    ) -> dict:
        """
        Verifica se a conversão manteve a precisão.
        
        Args:
            keras_model: Modelo Keras original
            tflite_path: Caminho do modelo TFLite
            test_input: Dados de teste
            tolerance: Tolerância para diferença nas predições
        
        Returns:
            Dicionário com métricas de comparação
        """
        # Predições do modelo Keras
        if len(test_input.shape) == 2:
            test_input_keras = test_input.reshape(-1, test_input.shape[1], 1)
        else:
            test_input_keras = test_input
        
        keras_predictions = keras_model.predict(test_input_keras, verbose=0)
        
        # Predições do modelo TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        tflite_predictions = []
        
        for sample in test_input_keras:
            interpreter.set_tensor(
                input_details[0]['index'],
                sample[np.newaxis, :, :].astype(np.float32)
            )
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            tflite_predictions.append(output[0])
        
        tflite_predictions = np.array(tflite_predictions)
        
        # Comparar
        diff = np.abs(keras_predictions.flatten() - tflite_predictions.flatten())
        
        return {
            'max_diff': float(np.max(diff)),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff)),
            'within_tolerance': bool(np.all(diff < tolerance)),
            'keras_mean': float(np.mean(keras_predictions)),
            'tflite_mean': float(np.mean(tflite_predictions))
        }


class TFLiteInference:
    """
    Classe para inferência com modelos TensorFlow Lite.
    
    Otimizada para baixa latência em dispositivos móveis.
    
    Exemplo de Uso:
        >>> inference = TFLiteInference('model.tflite')
        >>> heart_rate = inference.predict(ppg_signal)
    """
    
    def __init__(self, model_path: str, num_threads: int = 4):
        """
        Inicializa o executor de inferência.
        
        Args:
            model_path: Caminho do modelo .tflite
            num_threads: Número de threads para CPU
        """
        self.model_path = model_path
        
        # Criar interpreter
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Obter detalhes de entrada/saída
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Shape esperado
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"Modelo TFLite carregado: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Input dtype: {self.input_dtype}")
    
    def predict(self, signal: np.ndarray) -> float:
        """
        Faz predição de frequência cardíaca.
        
        Args:
            signal: Sinal PPG (array 1D ou 2D)
        
        Returns:
            Frequência cardíaca estimada em BPM
        """
        # Preparar input
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1, 1)
        elif len(signal.shape) == 2:
            signal = signal.reshape(signal.shape[0], -1, 1)
        
        # Converter para dtype correto
        signal = signal.astype(self.input_dtype)
        
        # Definir input
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            signal
        )
        
        # Executar
        self.interpreter.invoke()
        
        # Obter output
        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return float(output[0])
    
    def predict_batch(self, signals: np.ndarray) -> np.ndarray:
        """
        Faz predições em batch.
        
        Args:
            signals: Array de sinais PPG
        
        Returns:
            Array de frequências cardíacas
        """
        predictions = []
        
        for signal in signals:
            pred = self.predict(signal)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def benchmark(self, signal: np.ndarray, n_runs: int = 100) -> dict:
        """
        Mede performance da inferência.
        
        Args:
            signal: Sinal PPG de teste
            n_runs: Número de execuções
        
        Returns:
            Estatísticas de tempo
        """
        import time
        
        times = []
        
        # Warmup
        for _ in range(10):
            self.predict(signal)
        
        # Medir
        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(signal)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'fps': float(1000 / np.mean(times))
        }


def optimize_for_mobile(
    model: keras.Model,
    output_dir: str,
    sample_data: np.ndarray
) -> dict:
    """
    Otimiza modelo para diferentes cenários mobile.
    
    Gera múltiplas versões do modelo:
    - Sem quantização (baseline)
    - Quantização dinâmica (balanceado)
    - Float16 (GPU mobile)
    - Int8 (máxima eficiência)
    
    Args:
        model: Modelo Keras treinado
        output_dir: Diretório de saída
        sample_data: Dados de exemplo para calibração
    
    Returns:
        Dicionário com caminhos e métricas
    """
    os.makedirs(output_dir, exist_ok=True)
    converter = TFLiteConverter()
    
    results = {}
    
    # Versão sem quantização
    print("\n--- Convertendo sem quantização ---")
    path_none = os.path.join(output_dir, 'model_float32.tflite')
    converter.convert(model, path_none, quantization='none')
    results['float32'] = {
        'path': path_none,
        'size_kb': os.path.getsize(path_none) / 1024
    }
    
    # Quantização dinâmica
    print("\n--- Convertendo com quantização dinâmica ---")
    path_dynamic = os.path.join(output_dir, 'model_dynamic.tflite')
    converter.convert(model, path_dynamic, quantization='dynamic')
    results['dynamic'] = {
        'path': path_dynamic,
        'size_kb': os.path.getsize(path_dynamic) / 1024
    }
    
    # Float16
    print("\n--- Convertendo com float16 ---")
    path_fp16 = os.path.join(output_dir, 'model_float16.tflite')
    converter.convert(model, path_fp16, quantization='float16')
    results['float16'] = {
        'path': path_fp16,
        'size_kb': os.path.getsize(path_fp16) / 1024
    }
    
    # Int8 (se sample_data disponível)
    if sample_data is not None:
        print("\n--- Convertendo com int8 ---")
        path_int8 = os.path.join(output_dir, 'model_int8.tflite')
        rep_dataset = converter.create_representative_dataset(sample_data)
        try:
            converter.convert(
                model, path_int8,
                quantization='int8',
                representative_dataset=rep_dataset
            )
            results['int8'] = {
                'path': path_int8,
                'size_kb': os.path.getsize(path_int8) / 1024
            }
        except Exception as e:
            print(f"Erro na conversão int8: {e}")
    
    # Comparar tamanhos
    print("\n--- Comparação de tamanhos ---")
    for name, info in results.items():
        print(f"{name}: {info['size_kb']:.2f} KB")
    
    return results


if __name__ == "__main__":
    # Teste do módulo
    print("Testando conversão TFLite...")
    
    # Criar modelo simples para teste
    model = keras.Sequential([
        keras.layers.Input(shape=(300, 1)),
        keras.layers.Conv1D(16, 5, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 3, activation='relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Dados de teste
    X_test = np.random.randn(10, 300).astype(np.float32)
    
    # Converter
    converter = TFLiteConverter()
    
    # Versão dinâmica
    output_path = 'models/saved/test_model.tflite'
    os.makedirs('models/saved', exist_ok=True)
    converter.convert(model, output_path, quantization='dynamic')
    
    # Verificar conversão
    print("\n--- Verificando conversão ---")
    metrics = converter.verify_conversion(model, output_path, X_test)
    print(f"Métricas de verificação: {metrics}")
    
    # Testar inferência
    print("\n--- Testando inferência ---")
    inference = TFLiteInference(output_path)
    
    prediction = inference.predict(X_test[0])
    print(f"Predição: {prediction:.2f} BPM")
    
    # Benchmark
    print("\n--- Benchmark ---")
    benchmark = inference.benchmark(X_test[0])
    print(f"Performance: {benchmark}")
    
    print("\nTestes concluídos!")
