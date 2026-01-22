"""
=============================================================================
Modelos PyTorch para Estimativa de Frequência Cardíaca
=============================================================================
Este módulo implementa modelos de Deep Learning em PyTorch para processar
sinais PPG e estimar a frequência cardíaca.

Inclui:
- CNN-1D
- LSTM/GRU
- Modelo Híbrido CNN-LSTM
- Transformer simplificado

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Optional, List, Dict
import os


# ===========================================================================
# Dataset para PPG
# ===========================================================================

class PPGDataset(Dataset):
    """
    Dataset PyTorch para sinais PPG.
    
    Prepara os dados para treinamento/avaliação dos modelos.
    """
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        """
        Inicializa o dataset.
        
        Args:
            signals: Array de sinais PPG (n_samples, signal_length)
            labels: Array de frequências cardíacas (n_samples,)
        """
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.FloatTensor(labels)
        
        # Adicionar dimensão de canal se necessário
        if len(self.signals.shape) == 2:
            self.signals = self.signals.unsqueeze(1)  # (N, 1, L)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


# ===========================================================================
# Modelos CNN
# ===========================================================================

class HeartRateCNN1D(nn.Module):
    """
    Rede Neural Convolucional 1D em PyTorch.
    
    Arquitetura:
    Conv1D blocks -> Global Average Pooling -> Dense layers -> Output
    """
    
    def __init__(
        self,
        input_length: int = 300,
        conv_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [5, 5, 3],
        dense_units: List[int] = [128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Inicializa o modelo CNN.
        
        Args:
            input_length: Comprimento do sinal de entrada
            conv_channels: Número de canais para cada camada conv
            kernel_sizes: Tamanho do kernel para cada camada conv
            dense_units: Número de neurônios para camadas densas
            dropout_rate: Taxa de dropout
        """
        super(HeartRateCNN1D, self).__init__()
        
        self.input_length = input_length
        
        # Camadas convolucionais
        conv_layers = []
        in_channels = 1
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Camadas densas
        dense_layers = []
        in_features = conv_channels[-1]
        
        for units in dense_units:
            dense_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = units
        
        self.dense_layers = nn.Sequential(*dense_layers)
        
        # Camada de saída
        self.output_layer = nn.Linear(dense_units[-1], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada (batch, 1, length)
        
        Returns:
            Predições de frequência cardíaca (batch, 1)
        """
        # Convoluções
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Camadas densas
        x = self.dense_layers(x)
        
        # Saída
        x = self.output_layer(x)
        
        return x.squeeze(-1)


class HeartRateCNNLite(nn.Module):
    """
    Versão leve do modelo CNN para dispositivos móveis.
    """
    
    def __init__(self, input_length: int = 300):
        super(HeartRateCNNLite, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze(-1)


# ===========================================================================
# Modelos LSTM/GRU
# ===========================================================================

class HeartRateLSTM(nn.Module):
    """
    Modelo LSTM para estimativa de frequência cardíaca.
    """
    
    def __init__(
        self,
        input_length: int = 300,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.2
    ):
        super(HeartRateLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Tamanho da saída do LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, length) -> (batch, length, 1)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Usar último estado oculto
        if self.lstm.bidirectional:
            # Concatenar estados forward e backward
            h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_n = h_n[-1, :, :]
        
        # Camadas densas
        out = self.fc(h_n)
        
        return out.squeeze(-1)


class HeartRateGRU(nn.Module):
    """
    Modelo GRU para estimativa de frequência cardíaca.
    """
    
    def __init__(
        self,
        input_length: int = 300,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        super(HeartRateGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        gru_out, h_n = self.gru(x)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.fc(h_n)
        return out.squeeze(-1)


# ===========================================================================
# Modelo Híbrido CNN-LSTM
# ===========================================================================

class HeartRateCNNLSTM(nn.Module):
    """
    Modelo híbrido CNN-LSTM.
    
    - CNN extrai features locais
    - LSTM modela dependências temporais
    """
    
    def __init__(
        self,
        input_length: int = 300,
        conv_channels: List[int] = [32, 64],
        lstm_hidden: int = 32,
        dropout_rate: float = 0.3
    ):
        super(HeartRateCNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], 5, padding=2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], 3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Preparar para LSTM: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # Output
        out = self.fc(h_n)
        return out.squeeze(-1)


# ===========================================================================
# Trainer
# ===========================================================================

class HeartRateTrainer:
    """
    Classe para treinar e avaliar modelos PyTorch.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = 'auto'
    ):
        """
        Inicializa o trainer.
        
        Args:
            model: Modelo PyTorch a treinar
            learning_rate: Taxa de aprendizado
            device: Dispositivo ('cpu', 'cuda', ou 'auto')
        """
        # Determinar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Treina o modelo.
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação (opcional)
            epochs: Número de épocas
            early_stopping_patience: Paciência para early stopping
            verbose: Se True, imprime progresso
        
        Returns:
            Histórico do treinamento
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Treinamento
            self.model.train()
            train_loss = 0.0
            
            for signals, labels in train_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validação
            if val_loader is not None:
                val_loss, val_mae = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)
                
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Val MAE: {val_mae:.2f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Restaurar melhor modelo
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Valida o modelo."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
        
        return val_loss, mae
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        self.model.eval()
        
        # Preparar dados
        if len(X.shape) == 1:
            X = X.reshape(1, 1, -1)
        elif len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, -1)
        
        X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Avalia o modelo."""
        val_loss, mae = self._validate(test_loader)
        
        # Coletar todas as predições
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(self.device)
                outputs = self.model(signals)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        rmse = np.sqrt(np.mean((preds - labels) ** 2))
        mape = np.mean(np.abs((preds - labels) / labels)) * 100
        correlation = np.corrcoef(preds, labels)[0, 1]
        
        return {
            'loss': val_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'pearson_r': correlation
        }
    
    def save(self, filepath: str):
        """Salva o modelo."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load(self, filepath: str):
        """Carrega o modelo."""
        self.model.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )
        print(f"Modelo carregado de: {filepath}")


# ===========================================================================
# Funções utilitárias
# ===========================================================================

def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders para treinamento e validação.
    
    Args:
        X_train, y_train: Dados de treinamento
        X_val, y_val: Dados de validação
        batch_size: Tamanho do batch
    
    Returns:
        Tuple (train_loader, val_loader)
    """
    train_dataset = PPGDataset(X_train, y_train)
    val_dataset = PPGDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Teste do módulo
    print("Testando modelos PyTorch...")
    print(f"Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Criar dados sintéticos
    X = np.random.randn(100, 300).astype(np.float32)
    y = np.random.uniform(60, 100, 100).astype(np.float32)
    
    # Dividir dados
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    # Criar data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
    
    # Testar CNN
    print("\n--- HeartRateCNN1D ---")
    model = HeartRateCNN1D(input_length=300)
    print(model)
    
    # Treinar brevemente
    trainer = HeartRateTrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=5, verbose=True)
    
    # Avaliar
    metrics = trainer.evaluate(val_loader)
    print(f"\nMétricas: {metrics}")
    
    # Testar LSTM
    print("\n--- HeartRateLSTM ---")
    lstm = HeartRateLSTM(input_length=300)
    print(lstm)
    
    # Testar Híbrido
    print("\n--- HeartRateCNNLSTM ---")
    hybrid = HeartRateCNNLSTM(input_length=300)
    print(hybrid)
    
    print("\nTestes concluídos!")
