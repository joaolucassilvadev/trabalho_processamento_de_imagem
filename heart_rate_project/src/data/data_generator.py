"""
=============================================================================
Módulo de Geração de Dados para Treinamento
=============================================================================
Este módulo implementa funções para:
1. Gerar dados sintéticos de PPG para testes
2. Carregar e processar datasets públicos (UBFC-rPPG, PURE)
3. Extrair frames e sinais de vídeos

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import os
from dataclasses import dataclass
import json


@dataclass
class PPGSample:
    """Amostra de dados PPG."""
    signal: np.ndarray      # Sinal PPG
    heart_rate: float       # Frequência cardíaca (BPM)
    subject_id: str         # ID do sujeito
    quality: float = 1.0    # Qualidade do sinal (0-1)


class SyntheticPPGGenerator:
    """
    Gerador de sinais PPG sintéticos para treinamento e testes.
    
    Gera sinais realistas com:
    - Forma de onda PPG típica
    - Variabilidade de frequência cardíaca
    - Ruído e artefatos configuráveis
    - Modulação respiratória
    
    Útil para:
    - Testes unitários
    - Aumento de dados (data augmentation)
    - Validação de algoritmos
    
    Exemplo:
        >>> generator = SyntheticPPGGenerator()
        >>> signals, labels = generator.generate(n_samples=1000)
    """
    
    def __init__(
        self,
        sampling_rate: float = 30.0,
        signal_length: int = 300,
        heart_rate_range: Tuple[float, float] = (50, 120),
        noise_level: float = 0.1,
        respiratory_rate: float = 0.25
    ):
        """
        Inicializa o gerador.
        
        Args:
            sampling_rate: Taxa de amostragem em Hz
            signal_length: Número de amostras por sinal
            heart_rate_range: Faixa de frequência cardíaca (min, max) em BPM
            noise_level: Nível de ruído (0-1)
            respiratory_rate: Frequência respiratória em Hz (~0.25 = 15 resp/min)
        """
        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        self.heart_rate_range = heart_rate_range
        self.noise_level = noise_level
        self.respiratory_rate = respiratory_rate
    
    def generate_single(
        self,
        heart_rate: Optional[float] = None,
        add_noise: bool = True,
        add_artifacts: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Gera um único sinal PPG sintético.
        
        Args:
            heart_rate: Frequência cardíaca desejada (None = aleatória)
            add_noise: Se True, adiciona ruído gaussiano
            add_artifacts: Se True, adiciona artefatos de movimento
        
        Returns:
            Tuple (sinal_ppg, frequência_cardíaca)
        """
        # Definir frequência cardíaca
        if heart_rate is None:
            heart_rate = np.random.uniform(*self.heart_rate_range)
        
        # Tempo
        t = np.arange(self.signal_length) / self.sampling_rate
        
        # Frequência fundamental em Hz
        f0 = heart_rate / 60.0
        
        # Gerar forma de onda PPG
        # Componente sistólico (pico principal)
        systolic = self._generate_systolic_wave(t, f0)
        
        # Componente diastólico (pico secundário)
        diastolic = self._generate_diastolic_wave(t, f0)
        
        # Combinar
        ppg = systolic + 0.3 * diastolic
        
        # Adicionar modulação respiratória
        ppg = self._add_respiratory_modulation(ppg, t)
        
        # Adicionar variabilidade de frequência cardíaca (HRV)
        ppg = self._add_hrv(ppg, t, f0)
        
        # Adicionar ruído
        if add_noise:
            ppg = self._add_noise(ppg)
        
        # Adicionar artefatos de movimento
        if add_artifacts:
            ppg = self._add_artifacts(ppg)
        
        # Normalizar
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-6)
        
        return ppg, heart_rate
    
    def _generate_systolic_wave(
        self,
        t: np.ndarray,
        frequency: float
    ) -> np.ndarray:
        """Gera componente sistólico da onda PPG."""
        # Forma de onda usando seno com harmônicos
        wave = (
            np.sin(2 * np.pi * frequency * t) +
            0.5 * np.sin(2 * np.pi * 2 * frequency * t - np.pi/4) +
            0.25 * np.sin(2 * np.pi * 3 * frequency * t - np.pi/3)
        )
        return wave
    
    def _generate_diastolic_wave(
        self,
        t: np.ndarray,
        frequency: float
    ) -> np.ndarray:
        """Gera componente diastólico (notch dicrótico)."""
        # Onda deslocada em fase
        phase_shift = 0.3 * (1 / frequency)  # ~30% do ciclo
        wave = np.sin(2 * np.pi * frequency * (t - phase_shift))
        return wave
    
    def _add_respiratory_modulation(
        self,
        signal: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """Adiciona modulação respiratória ao sinal."""
        # Modulação de amplitude
        resp_modulation = 1 + 0.2 * np.sin(2 * np.pi * self.respiratory_rate * t)
        return signal * resp_modulation
    
    def _add_hrv(
        self,
        signal: np.ndarray,
        t: np.ndarray,
        base_freq: float
    ) -> np.ndarray:
        """Adiciona variabilidade de frequência cardíaca."""
        # Variação de frequência de baixa frequência
        freq_variation = 0.05 * np.sin(2 * np.pi * 0.1 * t) * base_freq
        
        # Modular fase do sinal
        phase = np.cumsum(freq_variation / self.sampling_rate) * 2 * np.pi
        modulated = signal * np.cos(phase * 0.5)
        
        return signal + 0.1 * modulated
    
    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona ruído gaussiano."""
        noise = np.random.normal(0, self.noise_level, len(signal))
        return signal + noise
    
    def _add_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona artefatos de movimento."""
        # Número de artefatos
        n_artifacts = np.random.randint(1, 4)
        
        for _ in range(n_artifacts):
            # Posição do artefato
            pos = np.random.randint(0, len(signal) - 30)
            duration = np.random.randint(10, 30)
            
            # Tipo de artefato
            artifact_type = np.random.choice(['spike', 'baseline_shift', 'noise_burst'])
            
            if artifact_type == 'spike':
                signal[pos:pos+3] += np.random.uniform(2, 5) * np.random.choice([-1, 1])
            elif artifact_type == 'baseline_shift':
                shift = np.random.uniform(-2, 2)
                signal[pos:pos+duration] += np.linspace(0, shift, duration)
            else:
                signal[pos:pos+duration] += np.random.normal(0, 0.5, duration)
        
        return signal
    
    def generate(
        self,
        n_samples: int = 1000,
        add_noise: bool = True,
        add_artifacts: bool = False,
        stratified: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera múltiplas amostras de sinais PPG.
        
        Args:
            n_samples: Número de amostras a gerar
            add_noise: Se True, adiciona ruído
            add_artifacts: Se True, adiciona artefatos
            stratified: Se True, distribui HR uniformemente
        
        Returns:
            Tuple (sinais, frequências_cardíacas)
        """
        signals = []
        heart_rates = []
        
        if stratified:
            # Distribuir HR uniformemente na faixa
            hr_values = np.linspace(
                self.heart_rate_range[0],
                self.heart_rate_range[1],
                n_samples
            )
            np.random.shuffle(hr_values)
        else:
            hr_values = [None] * n_samples
        
        for hr in hr_values:
            signal, actual_hr = self.generate_single(
                heart_rate=hr,
                add_noise=add_noise,
                add_artifacts=add_artifacts
            )
            signals.append(signal)
            heart_rates.append(actual_hr)
        
        return np.array(signals), np.array(heart_rates)
    
    def generate_with_augmentation(
        self,
        n_base_samples: int = 500,
        augmentation_factor: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera dados com aumento (data augmentation).
        
        Técnicas de aumento:
        - Variação de ruído
        - Deslocamento temporal
        - Escala de amplitude
        - Adição de artefatos
        
        Args:
            n_base_samples: Número de amostras base
            augmentation_factor: Fator de aumento
        
        Returns:
            Tuple (sinais_aumentados, frequências_cardíacas)
        """
        # Gerar amostras base
        base_signals, base_hrs = self.generate(n_base_samples, add_noise=False)
        
        all_signals = list(base_signals)
        all_hrs = list(base_hrs)
        
        for signal, hr in zip(base_signals, base_hrs):
            for _ in range(augmentation_factor - 1):
                augmented = signal.copy()
                
                # Aplicar aumentos aleatórios
                if np.random.random() > 0.5:
                    # Variação de ruído
                    noise_level = np.random.uniform(0.05, 0.2)
                    augmented += np.random.normal(0, noise_level, len(signal))
                
                if np.random.random() > 0.5:
                    # Deslocamento temporal
                    shift = np.random.randint(-20, 20)
                    augmented = np.roll(augmented, shift)
                
                if np.random.random() > 0.5:
                    # Escala de amplitude
                    scale = np.random.uniform(0.8, 1.2)
                    augmented *= scale
                
                if np.random.random() > 0.3:
                    # Artefatos leves
                    augmented = self._add_artifacts(augmented)
                
                all_signals.append(augmented)
                all_hrs.append(hr)
        
        # Embaralhar
        indices = np.random.permutation(len(all_signals))
        
        return np.array(all_signals)[indices], np.array(all_hrs)[indices]


class UBFCDatasetLoader:
    """
    Carregador para o dataset UBFC-rPPG.
    
    O UBFC-rPPG contém:
    - Vídeos de 42 sujeitos
    - Ground truth de PPG e frequência cardíaca
    - Formato: vid.avi + ground_truth.txt
    
    Download: https://sites.google.com/view/ybenezeth/ubfcrppg
    """
    
    def __init__(self, dataset_path: str):
        """
        Inicializa o carregador.
        
        Args:
            dataset_path: Caminho para o diretório UBFC-rPPG
        """
        self.dataset_path = dataset_path
        self.subjects = self._find_subjects()
    
    def _find_subjects(self) -> List[str]:
        """Encontra todos os sujeitos no dataset."""
        if not os.path.exists(self.dataset_path):
            print(f"Aviso: Dataset não encontrado em {self.dataset_path}")
            return []
        
        subjects = []
        for item in os.listdir(self.dataset_path):
            subject_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(subject_path):
                # Verificar se tem os arquivos necessários
                vid_path = os.path.join(subject_path, 'vid.avi')
                gt_path = os.path.join(subject_path, 'ground_truth.txt')
                
                if os.path.exists(vid_path) and os.path.exists(gt_path):
                    subjects.append(item)
        
        return sorted(subjects)
    
    def load_ground_truth(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega ground truth de um sujeito.
        
        Args:
            subject_id: ID do sujeito
        
        Returns:
            Tuple (timestamps, heart_rates)
        """
        gt_path = os.path.join(self.dataset_path, subject_id, 'ground_truth.txt')
        
        data = np.loadtxt(gt_path)
        
        # Formato: timestamp, heart_rate, (outros campos opcionais)
        if data.ndim == 1:
            # Apenas valores de HR
            timestamps = np.arange(len(data))
            heart_rates = data
        else:
            timestamps = data[:, 0]
            heart_rates = data[:, 1] if data.shape[1] > 1 else data[:, 0]
        
        return timestamps, heart_rates
    
    def extract_ppg_signal(
        self,
        subject_id: str,
        channel: str = 'green',
        roi_type: str = 'full_face',
        max_frames: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Extrai sinal PPG do vídeo de um sujeito.
        
        Args:
            subject_id: ID do sujeito
            channel: Canal de cor ('red', 'green', 'blue')
            roi_type: Tipo de ROI
            max_frames: Limite de frames (None = todos)
        
        Returns:
            Tuple (sinal_ppg, fps)
        """
        import cv2
        
        video_path = os.path.join(self.dataset_path, subject_id, 'vid.avi')
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames is None:
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        channel_map = {'blue': 0, 'green': 1, 'red': 2}
        ch_idx = channel_map.get(channel, 1)
        
        signal = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extrair valor médio do canal
            value = frame[:, :, ch_idx].mean()
            signal.append(value)
            frame_count += 1
        
        cap.release()
        
        return np.array(signal), fps
    
    def load_all_subjects(
        self,
        signal_length: int = 300,
        overlap: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega dados de todos os sujeitos.
        
        Args:
            signal_length: Comprimento das janelas de sinal
            overlap: Sobreposição entre janelas (0-1)
        
        Returns:
            Tuple (sinais, frequências_cardíacas)
        """
        all_signals = []
        all_hrs = []
        
        step = int(signal_length * (1 - overlap))
        
        for subject in self.subjects:
            print(f"Processando {subject}...")
            
            try:
                signal, fps = self.extract_ppg_signal(subject)
                _, heart_rates = self.load_ground_truth(subject)
                
                # Extrair janelas
                for start in range(0, len(signal) - signal_length, step):
                    window = signal[start:start + signal_length]
                    
                    # HR médio na janela
                    hr_start = int(start * len(heart_rates) / len(signal))
                    hr_end = int((start + signal_length) * len(heart_rates) / len(signal))
                    hr = np.mean(heart_rates[hr_start:hr_end])
                    
                    # Normalizar
                    window = (window - window.mean()) / (window.std() + 1e-6)
                    
                    all_signals.append(window)
                    all_hrs.append(hr)
                    
            except Exception as e:
                print(f"Erro ao processar {subject}: {e}")
        
        return np.array(all_signals), np.array(all_hrs)


def prepare_training_data(
    synthetic_samples: int = 5000,
    dataset_path: Optional[str] = None,
    signal_length: int = 300,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepara dados para treinamento.
    
    Args:
        synthetic_samples: Número de amostras sintéticas
        dataset_path: Caminho para dataset real (opcional)
        signal_length: Comprimento do sinal
        test_ratio: Proporção de teste
        val_ratio: Proporção de validação
    
    Returns:
        Dicionário com dados de treino, validação e teste
    """
    # Gerar dados sintéticos
    generator = SyntheticPPGGenerator(
        signal_length=signal_length,
        heart_rate_range=(50, 120)
    )
    
    X, y = generator.generate_with_augmentation(
        n_base_samples=synthetic_samples // 4,
        augmentation_factor=4
    )
    
    # Adicionar dados reais se disponíveis
    if dataset_path and os.path.exists(dataset_path):
        loader = UBFCDatasetLoader(dataset_path)
        X_real, y_real = loader.load_all_subjects(signal_length=signal_length)
        
        if len(X_real) > 0:
            X = np.concatenate([X, X_real])
            y = np.concatenate([y, y_real])
    
    # Embaralhar
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Dividir
    n = len(X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    X_test = X[:n_test]
    y_test = y[:n_test]
    
    X_val = X[n_test:n_test + n_val]
    y_val = y[n_test:n_test + n_val]
    
    X_train = X[n_test + n_val:]
    y_train = y[n_test + n_val:]
    
    print(f"\nDados preparados:")
    print(f"  Treino: {len(X_train)} amostras")
    print(f"  Validação: {len(X_val)} amostras")
    print(f"  Teste: {len(X_test)} amostras")
    print(f"  HR range: {y.min():.0f} - {y.max():.0f} BPM")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


if __name__ == "__main__":
    # Teste do módulo
    print("Testando geração de dados...")
    
    # Gerar dados sintéticos
    generator = SyntheticPPGGenerator()
    
    # Gerar amostra única
    signal, hr = generator.generate_single(heart_rate=72)
    print(f"\nAmostra única:")
    print(f"  Shape: {signal.shape}")
    print(f"  HR: {hr:.1f} BPM")
    
    # Gerar múltiplas amostras
    signals, hrs = generator.generate(n_samples=100)
    print(f"\nMúltiplas amostras:")
    print(f"  Shape: {signals.shape}")
    print(f"  HR range: {hrs.min():.1f} - {hrs.max():.1f} BPM")
    
    # Preparar dados de treinamento
    print("\n" + "="*50)
    data = prepare_training_data(synthetic_samples=1000)
    
    print("\nTeste concluído!")
