"""
=============================================================================
Módulo de Processamento de Sinais PPG
=============================================================================
Este módulo contém funções para processamento de sinais de fotopletismografia,
incluindo filtragem, normalização e extração de características.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks, detrend
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class PPGSignalProcessor:
    """
    Classe para processamento de sinais PPG (Fotopletismografia).
    
    Esta classe implementa um pipeline completo de processamento de sinais PPG,
    desde a filtragem até a estimativa de frequência cardíaca.
    
    Atributos:
        sampling_rate (float): Taxa de amostragem do sinal em Hz
        low_cutoff (float): Frequência de corte inferior do filtro passa-banda
        high_cutoff (float): Frequência de corte superior do filtro passa-banda
        filter_order (int): Ordem do filtro Butterworth
    
    Exemplo de Uso:
        >>> processor = PPGSignalProcessor(sampling_rate=30)
        >>> filtered_signal = processor.bandpass_filter(raw_signal)
        >>> heart_rate = processor.estimate_heart_rate(filtered_signal)
    """
    
    def __init__(
        self,
        sampling_rate: float = 30.0,
        low_cutoff: float = 0.7,
        high_cutoff: float = 4.0,
        filter_order: int = 4
    ):
        """
        Inicializa o processador de sinais PPG.
        
        Args:
            sampling_rate: Taxa de amostragem em Hz (padrão: 30 Hz para câmeras)
            low_cutoff: Frequência de corte inferior em Hz (0.7 Hz ≈ 42 BPM)
            high_cutoff: Frequência de corte superior em Hz (4.0 Hz ≈ 240 BPM)
            filter_order: Ordem do filtro Butterworth
        """
        self.sampling_rate = sampling_rate
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.filter_order = filter_order
        
        # Pré-calcular coeficientes do filtro para eficiência
        self._b, self._a = self._design_bandpass_filter()
    
    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projeta um filtro Butterworth passa-banda.
        
        O filtro passa-banda é essencial para isolar as frequências
        correspondentes aos batimentos cardíacos (tipicamente 0.7-4 Hz).
        
        Returns:
            Tuple contendo coeficientes do numerador (b) e denominador (a)
        """
        # Frequência de Nyquist (metade da taxa de amostragem)
        nyquist = self.sampling_rate / 2.0
        
        # Normalizar frequências de corte
        low = self.low_cutoff / nyquist
        high = self.high_cutoff / nyquist
        
        # Garantir que as frequências estão no intervalo válido
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        
        if low >= high:
            low = high * 0.5
        
        # Projetar filtro Butterworth
        b, a = butter(self.filter_order, [low, high], btype='band')
        
        return b, a
    
    def bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Aplica filtro passa-banda ao sinal PPG.
        
        O filtro remove componentes de baixa frequência (variações lentas
        devido a respiração, movimento) e alta frequência (ruído).
        
        Args:
            signal_data: Sinal PPG bruto como array numpy
        
        Returns:
            Sinal filtrado
        
        Raises:
            ValueError: Se o sinal for muito curto para filtragem
        """
        # Verificar comprimento mínimo do sinal
        min_length = 3 * self.filter_order
        if len(signal_data) < min_length:
            raise ValueError(
                f"Sinal muito curto. Mínimo: {min_length} amostras, "
                f"recebido: {len(signal_data)}"
            )
        
        # Aplicar filtro bidirecionalmente (zero-phase) para evitar
        # deslocamento de fase
        try:
            filtered = filtfilt(self._b, self._a, signal_data)
        except Exception as e:
            # Fallback: filtro simples se filtfilt falhar
            filtered = signal.lfilter(self._b, self._a, signal_data)
        
        return filtered
    
    def remove_trend(
        self,
        signal_data: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Remove tendência (trend) do sinal PPG.
        
        Variações lentas (baseline wander) podem ocorrer devido a
        mudanças de pressão, movimento ou iluminação.
        
        Args:
            signal_data: Sinal PPG
            method: Método de remoção ('linear' ou 'polynomial')
        
        Returns:
            Sinal sem tendência
        """
        if method == 'linear':
            return detrend(signal_data, type='linear')
        elif method == 'polynomial':
            # Ajuste polinomial de grau 3
            x = np.arange(len(signal_data))
            coeffs = np.polyfit(x, signal_data, 3)
            trend = np.polyval(coeffs, x)
            return signal_data - trend
        else:
            return signal_data
    
    def normalize(
        self,
        signal_data: np.ndarray,
        method: str = 'z-score'
    ) -> np.ndarray:
        """
        Normaliza o sinal PPG.
        
        A normalização é importante para que sinais de diferentes
        amplitudes sejam comparáveis.
        
        Args:
            signal_data: Sinal PPG
            method: Método de normalização ('z-score' ou 'min-max')
        
        Returns:
            Sinal normalizado
        """
        if method == 'z-score':
            # Normalização Z-score: (x - média) / desvio_padrão
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            if std > 0:
                return (signal_data - mean) / std
            return signal_data - mean
        
        elif method == 'min-max':
            # Normalização Min-Max: escala para [0, 1]
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            if max_val > min_val:
                return (signal_data - min_val) / (max_val - min_val)
            return np.zeros_like(signal_data)
        
        return signal_data
    
    def smooth(
        self,
        signal_data: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Suaviza o sinal usando média móvel.
        
        Args:
            signal_data: Sinal PPG
            window_size: Tamanho da janela de suavização
        
        Returns:
            Sinal suavizado
        """
        if window_size < 2:
            return signal_data
        
        # Média móvel usando convolução
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_data, kernel, mode='same')
        
        return smoothed
    
    def extract_ppg_from_rgb(
        self,
        frames: np.ndarray,
        channel: str = 'green',
        roi_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extrai sinal PPG de uma sequência de frames RGB.
        
        O sinal PPG é extraído calculando a média de intensidade
        na região de interesse (ROI) para cada frame.
        
        Para PPG com dedo: canal vermelho é mais eficaz (maior penetração)
        Para rPPG facial: canal verde é mais eficaz (absorção hemoglobina)
        
        Args:
            frames: Array de frames com shape (n_frames, height, width, 3)
            channel: Canal de cor a usar ('red', 'green', 'blue', ou 'all')
            roi_mask: Máscara binária opcional para região de interesse
        
        Returns:
            Sinal PPG como array 1D
        """
        channel_map = {'blue': 0, 'green': 1, 'red': 2}
        
        if len(frames.shape) != 4:
            raise ValueError("frames deve ter shape (n_frames, height, width, 3)")
        
        n_frames = frames.shape[0]
        ppg_signal = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = frames[i]
            
            # Aplicar máscara de ROI se fornecida
            if roi_mask is not None:
                frame = frame * roi_mask[:, :, np.newaxis]
                valid_pixels = roi_mask.sum()
            else:
                valid_pixels = frame.shape[0] * frame.shape[1]
            
            if valid_pixels == 0:
                continue
            
            # Extrair valor médio do canal especificado
            if channel == 'all':
                # Média ponderada dos três canais
                # Pesos baseados em estudos de rPPG
                weights = [0.2, 0.6, 0.2]  # B, G, R
                ppg_signal[i] = sum(
                    w * frame[:, :, c].sum() / valid_pixels
                    for c, w in enumerate(weights)
                )
            else:
                ch_idx = channel_map.get(channel, 1)  # Verde por padrão
                ppg_signal[i] = frame[:, :, ch_idx].sum() / valid_pixels
        
        return ppg_signal
    
    def compute_fft(
        self,
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula a Transformada Rápida de Fourier (FFT) do sinal.
        
        A FFT permite identificar as frequências dominantes no sinal,
        sendo a frequência fundamental correspondente à frequência cardíaca.
        
        Args:
            signal_data: Sinal PPG no domínio do tempo
        
        Returns:
            Tuple (frequências, magnitudes)
        """
        n = len(signal_data)
        
        # Aplicar janela de Hanning para reduzir vazamento espectral
        windowed = signal_data * np.hanning(n)
        
        # Calcular FFT
        fft_values = fft(windowed)
        frequencies = fftfreq(n, 1/self.sampling_rate)
        
        # Pegar apenas frequências positivas
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        magnitudes = np.abs(fft_values[positive_mask])
        
        # Normalizar magnitudes
        magnitudes = magnitudes / n
        
        return frequencies, magnitudes
    
    def estimate_heart_rate_fft(
        self,
        signal_data: np.ndarray,
        min_hr: float = 40,
        max_hr: float = 200
    ) -> Tuple[float, float]:
        """
        Estima frequência cardíaca usando análise FFT.
        
        O método encontra o pico de frequência dominante na faixa
        de frequências cardíacas válidas.
        
        Args:
            signal_data: Sinal PPG filtrado
            min_hr: Frequência cardíaca mínima esperada (BPM)
            max_hr: Frequência cardíaca máxima esperada (BPM)
        
        Returns:
            Tuple (frequência_cardíaca_bpm, confiança)
        """
        frequencies, magnitudes = self.compute_fft(signal_data)
        
        # Converter BPM para Hz
        min_freq = min_hr / 60.0
        max_freq = max_hr / 60.0
        
        # Filtrar frequências na faixa válida
        valid_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        valid_freqs = frequencies[valid_mask]
        valid_mags = magnitudes[valid_mask]
        
        if len(valid_mags) == 0:
            return 0.0, 0.0
        
        # Encontrar pico dominante
        peak_idx = np.argmax(valid_mags)
        peak_freq = valid_freqs[peak_idx]
        peak_mag = valid_mags[peak_idx]
        
        # Calcular frequência cardíaca em BPM
        heart_rate = peak_freq * 60.0
        
        # Calcular confiança baseada na razão sinal/ruído
        mean_mag = np.mean(valid_mags)
        if mean_mag > 0:
            confidence = min(peak_mag / mean_mag / 10.0, 1.0)
        else:
            confidence = 0.0
        
        return heart_rate, confidence
    
    def estimate_heart_rate_peaks(
        self,
        signal_data: np.ndarray,
        min_hr: float = 40,
        max_hr: float = 200
    ) -> Tuple[float, float]:
        """
        Estima frequência cardíaca usando detecção de picos.
        
        O método detecta picos no sinal PPG e calcula a frequência
        cardíaca baseada nos intervalos entre picos (IBI - Inter-Beat Interval).
        
        Args:
            signal_data: Sinal PPG filtrado
            min_hr: Frequência cardíaca mínima esperada (BPM)
            max_hr: Frequência cardíaca máxima esperada (BPM)
        
        Returns:
            Tuple (frequência_cardíaca_bpm, confiança)
        """
        # Calcular distância mínima e máxima entre picos
        min_distance = int(self.sampling_rate * 60 / max_hr)
        max_distance = int(self.sampling_rate * 60 / min_hr)
        
        # Detectar picos
        peaks, properties = find_peaks(
            signal_data,
            distance=min_distance,
            prominence=np.std(signal_data) * 0.5
        )
        
        if len(peaks) < 2:
            return 0.0, 0.0
        
        # Calcular intervalos entre picos
        peak_intervals = np.diff(peaks)
        
        # Filtrar intervalos válidos
        valid_intervals = peak_intervals[
            (peak_intervals >= min_distance) & 
            (peak_intervals <= max_distance)
        ]
        
        if len(valid_intervals) == 0:
            return 0.0, 0.0
        
        # Calcular frequência cardíaca média
        mean_interval = np.mean(valid_intervals)
        heart_rate = (self.sampling_rate / mean_interval) * 60
        
        # Calcular confiança baseada na variabilidade dos intervalos
        if len(valid_intervals) > 1:
            cv = np.std(valid_intervals) / np.mean(valid_intervals)
            confidence = max(0, 1 - cv)
        else:
            confidence = 0.5
        
        return heart_rate, confidence
    
    def estimate_heart_rate(
        self,
        signal_data: np.ndarray,
        method: str = 'combined',
        min_hr: float = 40,
        max_hr: float = 200
    ) -> Tuple[float, float]:
        """
        Estima frequência cardíaca usando método combinado.
        
        Combina estimativas de FFT e detecção de picos para maior robustez.
        
        Args:
            signal_data: Sinal PPG filtrado
            method: Método de estimativa ('fft', 'peaks', ou 'combined')
            min_hr: Frequência cardíaca mínima esperada (BPM)
            max_hr: Frequência cardíaca máxima esperada (BPM)
        
        Returns:
            Tuple (frequência_cardíaca_bpm, confiança)
        """
        if method == 'fft':
            return self.estimate_heart_rate_fft(signal_data, min_hr, max_hr)
        elif method == 'peaks':
            return self.estimate_heart_rate_peaks(signal_data, min_hr, max_hr)
        else:
            # Método combinado
            hr_fft, conf_fft = self.estimate_heart_rate_fft(
                signal_data, min_hr, max_hr
            )
            hr_peaks, conf_peaks = self.estimate_heart_rate_peaks(
                signal_data, min_hr, max_hr
            )
            
            # Média ponderada pela confiança
            total_conf = conf_fft + conf_peaks
            if total_conf > 0:
                heart_rate = (
                    hr_fft * conf_fft + hr_peaks * conf_peaks
                ) / total_conf
                confidence = total_conf / 2
            else:
                heart_rate = 0.0
                confidence = 0.0
            
            return heart_rate, confidence
    
    def process_pipeline(
        self,
        raw_signal: np.ndarray,
        detrend_method: str = 'linear',
        normalize_method: str = 'z-score'
    ) -> Tuple[np.ndarray, float, float]:
        """
        Executa pipeline completo de processamento.
        
        Pipeline:
        1. Remoção de tendência
        2. Filtragem passa-banda
        3. Normalização
        4. Suavização
        5. Estimativa de frequência cardíaca
        
        Args:
            raw_signal: Sinal PPG bruto
            detrend_method: Método de remoção de tendência
            normalize_method: Método de normalização
        
        Returns:
            Tuple (sinal_processado, frequência_cardíaca, confiança)
        """
        # 1. Remover tendência
        detrended = self.remove_trend(raw_signal, detrend_method)
        
        # 2. Filtrar
        filtered = self.bandpass_filter(detrended)
        
        # 3. Normalizar
        normalized = self.normalize(filtered, normalize_method)
        
        # 4. Suavizar
        smoothed = self.smooth(normalized, window_size=3)
        
        # 5. Estimar frequência cardíaca
        heart_rate, confidence = self.estimate_heart_rate(smoothed)
        
        return smoothed, heart_rate, confidence


class PPGQualityAssessor:
    """
    Classe para avaliação da qualidade do sinal PPG.
    
    Avalia se o sinal capturado é de qualidade suficiente para
    estimativa confiável de frequência cardíaca.
    """
    
    def __init__(self, sampling_rate: float = 30.0):
        """
        Inicializa o avaliador de qualidade.
        
        Args:
            sampling_rate: Taxa de amostragem do sinal
        """
        self.sampling_rate = sampling_rate
    
    def assess_signal_quality(
        self,
        signal_data: np.ndarray
    ) -> Tuple[float, dict]:
        """
        Avalia a qualidade do sinal PPG.
        
        Métricas avaliadas:
        - Razão Sinal-Ruído (SNR)
        - Periodicidade
        - Presença de artefatos
        
        Args:
            signal_data: Sinal PPG
        
        Returns:
            Tuple (score_qualidade, métricas_detalhadas)
        """
        metrics = {}
        
        # 1. Calcular SNR
        metrics['snr'] = self._compute_snr(signal_data)
        
        # 2. Avaliar periodicidade
        metrics['periodicity'] = self._assess_periodicity(signal_data)
        
        # 3. Detectar artefatos de movimento
        metrics['artifact_score'] = self._detect_artifacts(signal_data)
        
        # 4. Avaliar amplitude
        metrics['amplitude_score'] = self._assess_amplitude(signal_data)
        
        # Calcular score final (média ponderada)
        weights = {
            'snr': 0.3,
            'periodicity': 0.3,
            'artifact_score': 0.2,
            'amplitude_score': 0.2
        }
        
        quality_score = sum(
            metrics[k] * weights[k] for k in weights
        )
        
        return quality_score, metrics
    
    def _compute_snr(self, signal_data: np.ndarray) -> float:
        """Calcula razão sinal-ruído."""
        # Estimar sinal usando filtro passa-baixa
        b, a = butter(4, 0.5, btype='low')
        signal_estimate = filtfilt(b, a, signal_data)
        
        # Estimar ruído
        noise = signal_data - signal_estimate
        
        # Calcular SNR em dB
        signal_power = np.mean(signal_estimate ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            # Normalizar para [0, 1]
            return min(max(snr_db / 30, 0), 1)
        return 0.0
    
    def _assess_periodicity(self, signal_data: np.ndarray) -> float:
        """Avalia a periodicidade do sinal."""
        # Usar autocorrelação
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalizar
        
        # Encontrar primeiro pico (excluindo lag 0)
        min_lag = int(self.sampling_rate * 0.3)  # Mínimo 0.3s
        max_lag = int(self.sampling_rate * 2)    # Máximo 2s
        
        if max_lag <= min_lag or max_lag > len(autocorr):
            return 0.0
        
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.0
        
        peak_value = np.max(search_region)
        
        # Score baseado na força da periodicidade
        return max(0, min(peak_value, 1))
    
    def _detect_artifacts(self, signal_data: np.ndarray) -> float:
        """Detecta artefatos de movimento."""
        # Calcular derivada
        diff = np.diff(signal_data)
        
        # Detectar mudanças abruptas (artefatos)
        threshold = 3 * np.std(diff)
        artifacts = np.abs(diff) > threshold
        artifact_ratio = np.sum(artifacts) / len(diff)
        
        # Score: 1 = sem artefatos, 0 = muitos artefatos
        return max(0, 1 - artifact_ratio * 5)
    
    def _assess_amplitude(self, signal_data: np.ndarray) -> float:
        """Avalia amplitude do sinal."""
        amplitude = np.max(signal_data) - np.min(signal_data)
        
        # Amplitude muito baixa = sinal fraco
        # Amplitude muito alta = possível saturação
        if amplitude < 0.1:
            return amplitude / 0.1
        elif amplitude > 10:
            return max(0, 1 - (amplitude - 10) / 10)
        return 1.0


# =============================================================================
# Funções utilitárias
# =============================================================================

def create_synthetic_ppg(
    duration: float,
    sampling_rate: float,
    heart_rate: float,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Cria um sinal PPG sintético para testes.
    
    Args:
        duration: Duração do sinal em segundos
        sampling_rate: Taxa de amostragem em Hz
        heart_rate: Frequência cardíaca em BPM
        noise_level: Nível de ruído (0 a 1)
    
    Returns:
        Sinal PPG sintético
    """
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Frequência fundamental em Hz
    f0 = heart_rate / 60.0
    
    # Sinal PPG como soma de harmônicos
    ppg = (
        np.sin(2 * np.pi * f0 * t) +           # Fundamental
        0.5 * np.sin(2 * np.pi * 2 * f0 * t) + # 2º harmônico
        0.25 * np.sin(2 * np.pi * 3 * f0 * t)  # 3º harmônico
    )
    
    # Adicionar variação de amplitude (respiração)
    resp_freq = 0.25  # ~15 respirações por minuto
    ppg *= (1 + 0.3 * np.sin(2 * np.pi * resp_freq * t))
    
    # Adicionar ruído
    noise = np.random.normal(0, noise_level, len(t))
    ppg += noise
    
    return ppg


if __name__ == "__main__":
    # Exemplo de uso
    print("Testando módulo de processamento de sinais PPG...")
    
    # Criar sinal sintético
    ppg_signal = create_synthetic_ppg(
        duration=30,
        sampling_rate=30,
        heart_rate=72,
        noise_level=0.1
    )
    
    # Processar
    processor = PPGSignalProcessor(sampling_rate=30)
    processed, hr, confidence = processor.process_pipeline(ppg_signal)
    
    print(f"Frequência cardíaca estimada: {hr:.1f} BPM")
    print(f"Confiança: {confidence:.2%}")
    
    # Avaliar qualidade
    assessor = PPGQualityAssessor(sampling_rate=30)
    quality, metrics = assessor.assess_signal_quality(ppg_signal)
    
    print(f"Score de qualidade: {quality:.2%}")
    print(f"Métricas: {metrics}")
