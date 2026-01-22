"""
=============================================================================
Modelo LSTM para Estimativa de Frequência Cardíaca
=============================================================================
Este módulo implementa redes LSTM (Long Short-Term Memory) para processar
sinais PPG como séries temporais e estimar a frequência cardíaca.

O LSTM é especialmente adequado para sinais PPG por capturar dependências
temporais de longo prazo no sinal.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Optional, List, Dict, Tuple
import os


class HeartRateLSTM:
    """
    Modelo LSTM para estimativa de frequência cardíaca.
    
    Arquitetura:
    Input -> LSTM layers -> Dense layers -> Output
    
    As camadas LSTM processam o sinal PPG como uma sequência temporal,
    capturando padrões periódicos correspondentes aos batimentos cardíacos.
    
    Exemplo de Uso:
        >>> model = HeartRateLSTM(input_length=300)
        >>> model.build()
        >>> history = model.train(X_train, y_train, X_val, y_val)
    """
    
    def __init__(
        self,
        input_length: int = 300,
        lstm_units: List[int] = [64, 32],
        dense_units: List[int] = [32],
        dropout_rate: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001
    ):
        """
        Inicializa o modelo LSTM.
        
        Args:
            input_length: Número de amostras do sinal de entrada
            lstm_units: Lista com número de unidades para cada camada LSTM
            dense_units: Lista com número de neurônios para camadas densas
            dropout_rate: Taxa de dropout para regularização
            bidirectional: Se True, usa LSTM bidirecional
            learning_rate: Taxa de aprendizado inicial
        """
        self.input_length = input_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None
        self.history: Optional[dict] = None
    
    def build(self) -> Model:
        """
        Constrói a arquitetura do modelo LSTM.
        
        Returns:
            Modelo Keras compilado
        """
        inputs = layers.Input(shape=(self.input_length, 1), name='ppg_input')
        
        x = inputs
        
        # Camadas LSTM
        for i, units in enumerate(self.lstm_units):
            # Retornar sequências para todas menos a última LSTM
            return_sequences = (i < len(self.lstm_units) - 1)
            
            lstm_layer = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'lstm_{i+1}'
            )
            
            # Usar LSTM bidirecional se configurado
            if self.bidirectional:
                x = layers.Bidirectional(lstm_layer, name=f'bidirectional_{i+1}')(x)
            else:
                x = lstm_layer(x)
            
            # Batch normalization para estabilidade
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        
        # Camadas densas
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Saída - regressão
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='HeartRateLSTM')
        
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
        """
        Treina o modelo LSTM.
        
        Args:
            X_train: Dados de treinamento
            y_train: Labels de treinamento
            X_val: Dados de validação (opcional)
            y_val: Labels de validação (opcional)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            callbacks: Callbacks customizados
            verbose: Verbosidade
        
        Returns:
            Histórico do treinamento
        """
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
        """Retorna callbacks padrão para treinamento."""
        os.makedirs('models/saved', exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/saved/best_lstm_model.h5',
                monitor='val_mae',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        if self.model is None:
            raise ValueError("Modelo não construído")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(-1, self.input_length, 1)
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Avalia o modelo."""
        if self.model is None:
            raise ValueError("Modelo não construído")
        
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
            raise ValueError("Modelo não construído")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Carrega modelo."""
        self.model = keras.models.load_model(filepath)


class HeartRateGRU:
    """
    Modelo GRU (Gated Recurrent Unit) para estimativa de frequência cardíaca.
    
    GRU é uma alternativa mais leve ao LSTM com desempenho similar.
    """
    
    def __init__(
        self,
        input_length: int = 300,
        gru_units: List[int] = [64, 32],
        dropout_rate: float = 0.2
    ):
        self.input_length = input_length
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build(self) -> Model:
        """Constrói modelo GRU."""
        inputs = layers.Input(shape=(self.input_length, 1))
        
        x = inputs
        
        for i, units in enumerate(self.gru_units):
            return_sequences = (i < len(self.gru_units) - 1)
            
            x = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        self.model = Model(inputs, outputs, name='HeartRateGRU')
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.model


class HeartRateCNNLSTM:
    """
    Modelo híbrido CNN-LSTM para estimativa de frequência cardíaca.
    
    Combina:
    - CNN para extração de features locais
    - LSTM para modelagem de dependências temporais
    
    Esta arquitetura é eficaz para sinais PPG pois:
    1. CNN captura padrões morfológicos dos pulsos
    2. LSTM captura o ritmo e variabilidade entre pulsos
    """
    
    def __init__(
        self,
        input_length: int = 300,
        conv_filters: List[int] = [32, 64],
        kernel_sizes: List[int] = [5, 3],
        lstm_units: List[int] = [32],
        dropout_rate: float = 0.3
    ):
        self.input_length = input_length
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build(self) -> Model:
        """Constrói modelo híbrido CNN-LSTM."""
        inputs = layers.Input(shape=(self.input_length, 1))
        
        x = inputs
        
        # Blocos CNN para extração de features
        for filters, kernel in zip(self.conv_filters, self.kernel_sizes):
            x = layers.Conv1D(filters, kernel, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(2)(x)
        
        # Camadas LSTM para modelagem temporal
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            )(x)
        
        # Camadas de saída
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        self.model = Model(inputs, outputs, name='HeartRateCNNLSTM')
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def summary(self):
        if self.model is None:
            self.build()
        self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """Treina o modelo híbrido."""
        if self.model is None:
            self.build()
        
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(-1, self.input_length, 1)
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(-1, self.input_length, 1)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X.reshape(1, -1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(-1, self.input_length, 1)
        return self.model.predict(X, verbose=0).flatten()


class AttentionLayer(layers.Layer):
    """
    Camada de Atenção para LSTM.
    
    Permite que o modelo foque em partes específicas da sequência
    que são mais relevantes para a estimativa de frequência cardíaca.
    """
    
    def __init__(self, units: int = 64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Score = tanh(xW + b)
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        
        # Attention weights = softmax(score . u)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1),
            axis=1
        )
        
        # Context vector = sum(x * attention_weights)
        context = tf.reduce_sum(
            x * tf.expand_dims(attention_weights, -1),
            axis=1
        )
        
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class HeartRateLSTMAttention:
    """
    Modelo LSTM com mecanismo de Atenção.
    
    A atenção permite que o modelo aprenda quais partes do sinal
    são mais importantes para a estimativa de frequência cardíaca.
    """
    
    def __init__(
        self,
        input_length: int = 300,
        lstm_units: List[int] = [64, 64],
        attention_units: int = 32,
        dropout_rate: float = 0.3
    ):
        self.input_length = input_length
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build(self) -> Model:
        """Constrói modelo LSTM com atenção."""
        inputs = layers.Input(shape=(self.input_length, 1))
        
        x = inputs
        
        # Camadas LSTM (todas retornam sequências)
        for units in self.lstm_units:
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=True,
                    dropout=self.dropout_rate
                )
            )(x)
        
        # Camada de Atenção
        x = AttentionLayer(units=self.attention_units)(x)
        
        # Camadas densas
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        self.model = Model(inputs, outputs, name='HeartRateLSTMAttention')
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def summary(self):
        if self.model is None:
            self.build()
        self.model.summary()


if __name__ == "__main__":
    # Teste do módulo
    print("Testando modelos LSTM...")
    
    # Modelo LSTM básico
    print("\n--- HeartRateLSTM ---")
    model = HeartRateLSTM(input_length=300)
    model.build()
    model.summary()
    
    # Modelo GRU
    print("\n--- HeartRateGRU ---")
    gru = HeartRateGRU(input_length=300)
    gru.build()
    gru.model.summary()
    
    # Modelo Híbrido
    print("\n--- HeartRateCNNLSTM ---")
    hybrid = HeartRateCNNLSTM(input_length=300)
    hybrid.build()
    hybrid.summary()
    
    # Modelo com Atenção
    print("\n--- HeartRateLSTMAttention ---")
    attention = HeartRateLSTMAttention(input_length=300)
    attention.build()
    attention.summary()
    
    # Teste de predição
    X_test = np.random.randn(10, 300)
    predictions = model.predict(X_test)
    print(f"\nPredições de teste: {predictions[:5]}")
    
    print("\nTestes concluídos!")
