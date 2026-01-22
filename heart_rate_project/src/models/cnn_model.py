"""
=============================================================================
Modelo CNN-1D para Estimativa de Frequência Cardíaca
=============================================================================
Este módulo implementa uma Rede Neural Convolucional 1D para estimar
a frequência cardíaca a partir de sinais PPG.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, List, Dict
import os


class HeartRateCNN:
    """
    Rede Neural Convolucional 1D para estimativa de frequência cardíaca.
    
    Arquitetura:
    Input -> Conv1D blocks -> GlobalAvgPool -> Dense layers -> Output
    """
    
    def __init__(
        self,
        input_length: int = 300,
        conv_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [5, 5, 3],
        pool_sizes: List[int] = [2, 2, 2],
        dense_units: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Inicializa o modelo CNN.
        
        Args:
            input_length: Número de amostras do sinal de entrada
            conv_filters: Lista com número de filtros para cada camada conv
            kernel_sizes: Lista com tamanho do kernel para cada camada conv
            pool_sizes: Lista com tamanho do pooling para cada camada
            dense_units: Lista com número de neurônios para camadas densas
            dropout_rate: Taxa de dropout para regularização
            learning_rate: Taxa de aprendizado inicial
        """
        self.input_length = input_length
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None
        self.history: Optional[dict] = None
    
    def build(self) -> Model:
        """Constrói a arquitetura do modelo CNN."""
        inputs = layers.Input(shape=(self.input_length, 1), name='ppg_input')
        
        x = inputs
        
        # Blocos Convolucionais
        for i, (filters, kernel, pool) in enumerate(
            zip(self.conv_filters, self.kernel_sizes, self.pool_sizes)
        ):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel,
                padding='same',
                name=f'conv_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Activation('relu', name=f'relu_{i+1}')(x)
            x = layers.MaxPooling1D(pool_size=pool, name=f'pool_{i+1}')(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Camadas Densas
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
            x = layers.Activation('relu', name=f'dense_relu_{i+1}')(x)
        
        # Saída - regressão
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='HeartRateCNN')
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return self.model
    
    def summary(self):
        """Imprime resumo do modelo."""
        if self.model is None:
            self.build()
        self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> dict:
        """Treina o modelo."""
        if self.model is None:
            self.build()
        
        # Garantir shape correto
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(-1, self.input_length, 1)
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(-1, self.input_length, 1)
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        return self.history
    
    def _get_default_callbacks(self) -> List:
        """Retorna callbacks padrão."""
        os.makedirs('models/saved', exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/saved/best_cnn_model.h5',
                monitor='val_mae',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        if self.model is None:
            raise ValueError("Modelo não foi construído/carregado")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(-1, self.input_length, 1)
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Avalia o modelo."""
        if self.model is None:
            raise ValueError("Modelo não foi construído/carregado")
        
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(-1, self.input_length, 1)
        
        predictions = self.predict(X_test)
        
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'pearson_r': correlation
        }
    
    def save(self, filepath: str):
        """Salva o modelo."""
        if self.model is None:
            raise ValueError("Modelo não foi construído")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load(self, filepath: str):
        """Carrega modelo de arquivo."""
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")


class HeartRateCNNLite:
    """
    Versão leve do modelo CNN otimizada para dispositivos móveis.
    
    Usa menos parâmetros e é adequada para conversão TFLite.
    """
    
    def __init__(self, input_length: int = 300):
        self.input_length = input_length
        self.model = None
    
    def build(self) -> Model:
        """Constrói modelo leve."""
        inputs = layers.Input(shape=(self.input_length, 1))
        
        # Arquitetura simplificada
        x = layers.Conv1D(16, 7, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1)(x)
        
        self.model = Model(inputs, outputs, name='HeartRateCNNLite')
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.model


def create_residual_block(x, filters, kernel_size):
    """Cria um bloco residual para CNN."""
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Ajustar dimensões se necessário
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    
    return x


class HeartRateResNet:
    """
    Modelo ResNet-1D para estimativa de frequência cardíaca.
    
    Usa conexões residuais para melhor gradiente flow.
    """
    
    def __init__(self, input_length: int = 300):
        self.input_length = input_length
        self.model = None
    
    def build(self) -> Model:
        """Constrói modelo ResNet."""
        inputs = layers.Input(shape=(self.input_length, 1))
        
        # Camada inicial
        x = layers.Conv1D(32, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Blocos residuais
        x = create_residual_block(x, 32, 3)
        x = create_residual_block(x, 32, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = create_residual_block(x, 64, 3)
        x = create_residual_block(x, 64, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = create_residual_block(x, 128, 3)
        
        # Classificação
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)
        
        self.model = Model(inputs, outputs, name='HeartRateResNet')
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model


if __name__ == "__main__":
    # Teste do módulo
    print("Testando modelos CNN...")
    
    # Criar modelo padrão
    model = HeartRateCNN(input_length=300)
    model.build()
    model.summary()
    
    # Criar dados sintéticos para teste
    X_test = np.random.randn(100, 300)
    y_test = np.random.uniform(60, 100, 100)
    
    # Teste de predição (sem treinamento)
    predictions = model.predict(X_test[:5])
    print(f"\nPredições de teste: {predictions}")
    
    # Testar modelo leve
    print("\n--- Modelo Lite ---")
    lite_model = HeartRateCNNLite(input_length=300)
    lite_model.build()
    lite_model.model.summary()
    
    # Testar ResNet
    print("\n--- Modelo ResNet ---")
    resnet = HeartRateResNet(input_length=300)
    resnet.build()
    resnet.model.summary()
    
    print("\nTestes concluídos!")
