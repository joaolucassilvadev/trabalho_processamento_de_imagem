"""
=============================================================================
Módulo de Inferência em Tempo Real
=============================================================================
Este módulo implementa a captura de vídeo e inferência em tempo real
para estimativa de frequência cardíaca usando PPG.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, Callable, Dict
import time
import sys
import os
from dataclasses import dataclass
from enum import Enum

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from preprocessing.signal_processing import PPGSignalProcessor, PPGQualityAssessor
    from preprocessing.face_detector import FaceDetector, ROIExtractor, ROIType, FingerROIExtractor
except ImportError:
    # Fallback para execução standalone
    pass


class CaptureMode(Enum):
    FINGER = "finger"
    FACE = "face"


@dataclass
class HeartRateResult:
    """Resultado da estimativa de frequência cardíaca."""
    heart_rate: float
    confidence: float
    signal_quality: float
    timestamp: float
    mode: CaptureMode


class SimplePPGProcessor:
    """Processador simples de PPG para quando módulos não estão disponíveis."""
    
    def __init__(self, sampling_rate: float = 30.0):
        self.sampling_rate = sampling_rate
    
    def process(self, signal: np.ndarray) -> Tuple[float, float]:
        """Processa sinal e retorna HR e confiança."""
        from scipy import signal as sig
        from scipy.fft import fft, fftfreq
        
        if len(signal) < 30:
            return 0.0, 0.0
        
        # Remover tendência
        signal = sig.detrend(signal)
        
        # Filtro passa-banda
        nyq = self.sampling_rate / 2
        b, a = sig.butter(4, [0.7/nyq, 4.0/nyq], btype='band')
        filtered = sig.filtfilt(b, a, signal)
        
        # FFT para encontrar frequência dominante
        n = len(filtered)
        fft_vals = np.abs(fft(filtered))[:n//2]
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        # Filtrar para faixa de HR (40-200 BPM = 0.67-3.33 Hz)
        mask = (freqs >= 0.67) & (freqs <= 3.33)
        valid_freqs = freqs[mask]
        valid_fft = fft_vals[mask]
        
        if len(valid_fft) == 0:
            return 0.0, 0.0
        
        # Encontrar pico
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        heart_rate = peak_freq * 60
        
        # Confiança baseada em SNR
        mean_mag = np.mean(valid_fft)
        confidence = min(valid_fft[peak_idx] / (mean_mag * 10 + 1e-6), 1.0)
        
        return heart_rate, confidence


class HeartRateEstimator:
    """
    Sistema de estimativa de frequência cardíaca em tempo real.
    
    Exemplo:
        >>> estimator = HeartRateEstimator(mode='finger')
        >>> estimator.start_capture()
    """
    
    def __init__(
        self,
        mode: str = 'finger',
        camera_index: int = 0,
        fps: int = 30,
        buffer_seconds: float = 10.0,
        callback: Optional[Callable[[HeartRateResult], None]] = None
    ):
        self.mode = CaptureMode(mode)
        self.camera_index = camera_index
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.callback = callback
        
        # Processador de sinal
        self.processor = SimplePPGProcessor(sampling_rate=fps)
        
        # Buffers
        self.signal_buffer = {
            'red': deque(maxlen=self.buffer_size),
            'green': deque(maxlen=self.buffer_size),
            'blue': deque(maxlen=self.buffer_size),
        }
        
        # Estado
        self.is_running = False
        self.current_hr = 0.0
        self.current_confidence = 0.0
        
        # Face detector para modo facial
        self.face_cascade = None
        if self.mode == CaptureMode.FACE:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.cap = None
    
    def start_capture(self, show_video: bool = True):
        """Inicia captura e processamento."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir câmera {self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        print(f"Captura iniciada - Modo: {self.mode.value}")
        print("Pressione 'q' para sair, 'm' para mudar modo")
        
        try:
            self._capture_loop(show_video)
        finally:
            self.stop_capture()
    
    def _capture_loop(self, show_video: bool):
        """Loop principal."""
        update_interval = 1.0
        last_update = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extrair valores de cor
            color_values = self._extract_color_values(frame)
            
            if color_values:
                self.signal_buffer['red'].append(color_values['red'])
                self.signal_buffer['green'].append(color_values['green'])
                self.signal_buffer['blue'].append(color_values['blue'])
            
            # Atualizar HR
            if time.time() - last_update >= update_interval:
                if len(self.signal_buffer['red']) >= self.fps * 3:
                    self._update_heart_rate()
                    last_update = time.time()
            
            # Mostrar vídeo
            if show_video:
                frame = self._draw_overlay(frame)
                cv2.imshow('Heart Rate Monitor', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self._toggle_mode()
    
    def _extract_color_values(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Extrai valores médios de cor da ROI."""
        if self.mode == CaptureMode.FINGER:
            return self._extract_finger(frame)
        else:
            return self._extract_face(frame)
    
    def _extract_finger(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Extrai valores para modo dedo."""
        h, w = frame.shape[:2]
        
        # ROI central
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        roi = frame[margin_y:h-margin_y, margin_x:w-margin_x]
        
        # Verificar se dedo está presente (vermelho dominante, brilhante)
        red_mean = roi[:, :, 2].mean()
        green_mean = roi[:, :, 1].mean()
        
        if red_mean < 50 or red_mean < green_mean:
            return None
        
        return {
            'red': float(red_mean),
            'green': float(green_mean),
            'blue': float(roi[:, :, 0].mean())
        }
    
    def _extract_face(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Extrai valores para modo facial."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Maior face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # ROI da testa
        forehead_y = y + int(h * 0.05)
        forehead_h = int(h * 0.2)
        forehead_x = x + int(w * 0.25)
        forehead_w = int(w * 0.5)
        
        roi = frame[forehead_y:forehead_y+forehead_h, 
                   forehead_x:forehead_x+forehead_w]
        
        if roi.size == 0:
            return None
        
        # Desenhar ROI
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (forehead_x, forehead_y), 
                     (forehead_x+forehead_w, forehead_y+forehead_h),
                     (255, 0, 0), 2)
        
        return {
            'red': float(roi[:, :, 2].mean()),
            'green': float(roi[:, :, 1].mean()),
            'blue': float(roi[:, :, 0].mean())
        }
    
    def _update_heart_rate(self):
        """Atualiza estimativa de HR."""
        # Usar canal apropriado
        if self.mode == CaptureMode.FINGER:
            signal = np.array(self.signal_buffer['red'])
        else:
            signal = np.array(self.signal_buffer['green'])
        
        hr, confidence = self.processor.process(signal)
        
        if 40 < hr < 200:
            # Suavizar
            if self.current_hr > 0:
                self.current_hr = 0.3 * hr + 0.7 * self.current_hr
            else:
                self.current_hr = hr
            self.current_confidence = confidence
            
            # Callback
            if self.callback:
                result = HeartRateResult(
                    heart_rate=self.current_hr,
                    confidence=self.current_confidence,
                    signal_quality=confidence,
                    timestamp=time.time(),
                    mode=self.mode
                )
                self.callback(result)
    
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Desenha informações no frame."""
        h, w = frame.shape[:2]
        
        # Fundo
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Modo
        cv2.putText(frame, f"Modo: {self.mode.value.upper()}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # HR
        if self.current_hr > 0:
            hr_text = f"HR: {self.current_hr:.0f} BPM"
            color = (0, 255, 0) if self.current_confidence > 0.5 else (0, 255, 255)
        else:
            hr_text = "HR: --"
            color = (128, 128, 128)
        
        cv2.putText(frame, hr_text, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Confiança
        cv2.putText(frame, f"Conf: {self.current_confidence:.0%}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instruções
        if self.mode == CaptureMode.FINGER:
            instr = "Coloque o dedo sobre a camera"
        else:
            instr = "Mantenha o rosto visivel"
        
        cv2.putText(frame, instr, (w//2 - 120, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _toggle_mode(self):
        """Alterna modo."""
        if self.mode == CaptureMode.FINGER:
            self.mode = CaptureMode.FACE
            if self.face_cascade is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.mode = CaptureMode.FINGER
        
        # Limpar buffers
        for key in self.signal_buffer:
            self.signal_buffer[key].clear()
        
        self.current_hr = 0.0
        self.current_confidence = 0.0
        print(f"\nModo alterado para: {self.mode.value}")
    
    def stop_capture(self):
        """Para a captura."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nCaptura encerrada")


def run_demo(mode: str = 'finger', camera: int = 0):
    """Executa demonstração."""
    print("=" * 60)
    print("Sistema de Estimativa de Frequência Cardíaca")
    print("=" * 60)
    print(f"\nModo: {mode}")
    print(f"Câmera: {camera}")
    print("\nControles:")
    print("  'q' - Sair")
    print("  'm' - Alternar modo")
    print("=" * 60)
    
    def on_result(result: HeartRateResult):
        print(f"\rHR: {result.heart_rate:5.1f} BPM | "
              f"Conf: {result.confidence:4.0%}", end='', flush=True)
    
    estimator = HeartRateEstimator(
        mode=mode,
        camera_index=camera,
        callback=on_result
    )
    
    try:
        estimator.start_capture(show_video=True)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Heart Rate Monitor')
    parser.add_argument('--mode', '-m', default='finger', choices=['finger', 'face'])
    parser.add_argument('--camera', '-c', type=int, default=0)
    
    args = parser.parse_args()
    run_demo(mode=args.mode, camera=args.camera)
