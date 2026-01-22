"""
=============================================================================
Aplicação Principal - Monitor de Frequência Cardíaca
=============================================================================
Este módulo implementa a aplicação com interface gráfica para
demonstração do sistema de estimativa de frequência cardíaca.

Funcionalidades:
- Captura de vídeo em tempo real
- Visualização do sinal PPG
- Exibição de frequência cardíaca
- Alternância entre modos (dedo/face)
- Gravação de sessões

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
from collections import deque
import time
import sys
import os
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Verificar dependências de GUI
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QGroupBox, QProgressBar,
        QFrame, QSplitter
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    print("PyQt5 não instalado. Usando interface de linha de comando.")


class CaptureMode(Enum):
    FINGER = "finger"
    FACE = "face"


@dataclass
class HeartRateResult:
    heart_rate: float
    confidence: float
    signal_quality: float
    timestamp: float


class SimplePPGProcessor:
    """Processador simples de PPG."""
    
    def __init__(self, sampling_rate: float = 30.0):
        self.sampling_rate = sampling_rate
    
    def process(self, signal: np.ndarray) -> tuple:
        """Processa sinal e retorna HR e confiança."""
        from scipy import signal as sig
        from scipy.fft import fft, fftfreq
        
        if len(signal) < 30:
            return 0.0, 0.0
        
        # Processar
        signal = sig.detrend(signal)
        nyq = self.sampling_rate / 2
        b, a = sig.butter(4, [0.7/nyq, 4.0/nyq], btype='band')
        filtered = sig.filtfilt(b, a, signal)
        
        # FFT
        n = len(filtered)
        fft_vals = np.abs(fft(filtered))[:n//2]
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        mask = (freqs >= 0.67) & (freqs <= 3.33)
        valid_freqs = freqs[mask]
        valid_fft = fft_vals[mask]
        
        if len(valid_fft) == 0:
            return 0.0, 0.0
        
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        heart_rate = peak_freq * 60
        
        mean_mag = np.mean(valid_fft)
        confidence = min(valid_fft[peak_idx] / (mean_mag * 10 + 1e-6), 1.0)
        
        return heart_rate, confidence


class HeartRateMonitor:
    """Classe base para monitoramento de frequência cardíaca."""
    
    def __init__(self, camera_index: int = 0, fps: int = 30):
        self.camera_index = camera_index
        self.fps = fps
        self.buffer_size = fps * 10
        
        self.mode = CaptureMode.FINGER
        self.processor = SimplePPGProcessor(sampling_rate=fps)
        
        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.current_hr = 0.0
        self.current_confidence = 0.0
        
        self.cap = None
        self.is_running = False
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def start(self):
        """Inicia captura."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
    
    def stop(self):
        """Para captura."""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Processa um frame e retorna (frame_processado, valor_sinal)."""
        if self.mode == CaptureMode.FINGER:
            return self._process_finger(frame)
        else:
            return self._process_face(frame)
    
    def _process_finger(self, frame: np.ndarray) -> tuple:
        """Processa frame no modo dedo."""
        h, w = frame.shape[:2]
        margin = int(min(h, w) * 0.2)
        roi = frame[margin:h-margin, margin:w-margin]
        
        red_mean = roi[:, :, 2].mean()
        green_mean = roi[:, :, 1].mean()
        
        # Desenhar ROI
        cv2.rectangle(frame, (margin, margin), (w-margin, h-margin), (0, 255, 0), 2)
        
        if red_mean < 50 or red_mean < green_mean:
            return frame, None
        
        return frame, red_mean
    
    def _process_face(self, frame: np.ndarray) -> tuple:
        """Processa frame no modo face."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return frame, None
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # ROI da testa
        fy = y + int(h * 0.05)
        fh = int(h * 0.2)
        fx = x + int(w * 0.25)
        fw = int(w * 0.5)
        
        roi = frame[fy:fy+fh, fx:fx+fw]
        
        # Desenhar
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        
        if roi.size == 0:
            return frame, None
        
        return frame, roi[:, :, 1].mean()  # Canal verde
    
    def update_heart_rate(self) -> HeartRateResult:
        """Atualiza estimativa de HR."""
        signal = np.array(self.signal_buffer)
        
        if len(signal) < 90:  # Mínimo 3 segundos
            return HeartRateResult(0, 0, 0, time.time())
        
        hr, confidence = self.processor.process(signal)
        
        if 40 < hr < 200:
            if self.current_hr > 0:
                self.current_hr = 0.3 * hr + 0.7 * self.current_hr
            else:
                self.current_hr = hr
            self.current_confidence = confidence
        
        return HeartRateResult(
            self.current_hr,
            self.current_confidence,
            confidence,
            time.time()
        )
    
    def toggle_mode(self):
        """Alterna modo."""
        if self.mode == CaptureMode.FINGER:
            self.mode = CaptureMode.FACE
        else:
            self.mode = CaptureMode.FINGER
        
        self.signal_buffer.clear()
        self.current_hr = 0.0
        self.current_confidence = 0.0


def run_cli_app(camera: int = 0):
    """Executa aplicação de linha de comando."""
    print("="*60)
    print("Monitor de Frequência Cardíaca - CLI")
    print("="*60)
    print("Controles: 'q'=sair, 'm'=mudar modo")
    print("="*60)
    
    monitor = HeartRateMonitor(camera_index=camera)
    monitor.start()
    
    last_update = time.time()
    
    try:
        while monitor.is_running:
            ret, frame = monitor.cap.read()
            if not ret:
                break
            
            # Processar frame
            frame, signal_value = monitor.process_frame(frame)
            
            if signal_value is not None:
                monitor.signal_buffer.append(signal_value)
            
            # Atualizar HR a cada segundo
            if time.time() - last_update >= 1.0:
                result = monitor.update_heart_rate()
                last_update = time.time()
                
                print(f"\rHR: {result.heart_rate:5.1f} BPM | "
                      f"Conf: {result.confidence:4.0%} | "
                      f"Modo: {monitor.mode.value}   ", end='', flush=True)
            
            # Desenhar info no frame
            h, w = frame.shape[:2]
            
            # Fundo
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Texto
            cv2.putText(frame, f"Modo: {monitor.mode.value.upper()}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if monitor.current_hr > 0:
                hr_text = f"HR: {monitor.current_hr:.0f} BPM"
                color = (0, 255, 0) if monitor.current_confidence > 0.5 else (0, 255, 255)
            else:
                hr_text = "HR: --"
                color = (128, 128, 128)
            
            cv2.putText(frame, hr_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame, f"Conf: {monitor.current_confidence:.0%}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mostrar
            cv2.imshow('Heart Rate Monitor', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                monitor.toggle_mode()
                print(f"\nModo alterado para: {monitor.mode.value}")
    
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    
    finally:
        monitor.stop()
        cv2.destroyAllWindows()
        print("\nAplicação encerrada")


if HAS_PYQT:
    class VideoThread(QThread):
        """Thread para captura de vídeo."""
        frame_ready = pyqtSignal(np.ndarray)
        result_ready = pyqtSignal(HeartRateResult)
        
        def __init__(self, monitor: HeartRateMonitor):
            super().__init__()
            self.monitor = monitor
            self.running = False
            self.last_update = 0
        
        def run(self):
            self.monitor.start()
            self.running = True
            
            while self.running and self.monitor.is_running:
                ret, frame = self.monitor.cap.read()
                if not ret:
                    continue
                
                # Processar
                frame, signal_value = self.monitor.process_frame(frame)
                
                if signal_value is not None:
                    self.monitor.signal_buffer.append(signal_value)
                
                # Emitir frame
                self.frame_ready.emit(frame)
                
                # Atualizar HR
                if time.time() - self.last_update >= 1.0:
                    result = self.monitor.update_heart_rate()
                    self.result_ready.emit(result)
                    self.last_update = time.time()
                
                time.sleep(1/30)  # ~30 FPS
        
        def stop(self):
            self.running = False
            self.monitor.stop()

    class MainWindow(QMainWindow):
        """Janela principal da aplicação."""
        
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("Monitor de Frequência Cardíaca")
            self.setMinimumSize(900, 600)
            
            # Estilo escuro
            self.setStyleSheet("""
                QMainWindow { background-color: #1e1e1e; }
                QLabel { color: #ffffff; }
                QPushButton {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #555555;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #4c4c4c; }
                QPushButton:pressed { background-color: #2c2c2c; }
                QGroupBox {
                    color: #ffffff;
                    border: 1px solid #555555;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QComboBox {
                    background-color: #3c3c3c;
                    color: #ffffff;
                    border: 1px solid #555555;
                    padding: 5px;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #00aa00;
                }
            """)
            
            # Monitor
            self.monitor = HeartRateMonitor()
            self.video_thread = None
            
            self._setup_ui()
        
        def _setup_ui(self):
            """Configura interface."""
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QHBoxLayout(central)
            
            # Painel de vídeo
            video_panel = QGroupBox("Câmera")
            video_layout = QVBoxLayout(video_panel)
            
            self.video_label = QLabel()
            self.video_label.setMinimumSize(640, 480)
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setStyleSheet("background-color: #000000;")
            video_layout.addWidget(self.video_label)
            
            layout.addWidget(video_panel, stretch=2)
            
            # Painel de controle
            control_panel = QWidget()
            control_layout = QVBoxLayout(control_panel)
            
            # Grupo de medições
            measurements = QGroupBox("Medições")
            measurements_layout = QVBoxLayout(measurements)
            
            self.hr_label = QLabel("-- BPM")
            self.hr_label.setFont(QFont("Arial", 48, QFont.Bold))
            self.hr_label.setAlignment(Qt.AlignCenter)
            self.hr_label.setStyleSheet("color: #00ff00;")
            measurements_layout.addWidget(self.hr_label)
            
            self.confidence_label = QLabel("Confiança: --%")
            self.confidence_label.setAlignment(Qt.AlignCenter)
            measurements_layout.addWidget(self.confidence_label)
            
            self.quality_bar = QProgressBar()
            self.quality_bar.setMaximum(100)
            self.quality_bar.setFormat("Qualidade: %v%")
            measurements_layout.addWidget(self.quality_bar)
            
            control_layout.addWidget(measurements)
            
            # Grupo de controles
            controls = QGroupBox("Controles")
            controls_layout = QVBoxLayout(controls)
            
            mode_layout = QHBoxLayout()
            mode_layout.addWidget(QLabel("Modo:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Dedo", "Face"])
            self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            mode_layout.addWidget(self.mode_combo)
            controls_layout.addLayout(mode_layout)
            
            self.start_btn = QPushButton("Iniciar")
            self.start_btn.clicked.connect(self._toggle_capture)
            controls_layout.addWidget(self.start_btn)
            
            control_layout.addWidget(controls)
            
            # Instruções
            instructions = QGroupBox("Instruções")
            instructions_layout = QVBoxLayout(instructions)
            
            self.instruction_label = QLabel(
                "Modo Dedo:\n"
                "Coloque o dedo sobre a câmera\n"
                "com o flash ligado.\n\n"
                "Modo Face:\n"
                "Posicione o rosto na frente\n"
                "da câmera e mantenha-se parado."
            )
            self.instruction_label.setWordWrap(True)
            instructions_layout.addWidget(self.instruction_label)
            
            control_layout.addWidget(instructions)
            control_layout.addStretch()
            
            layout.addWidget(control_panel, stretch=1)
        
        def _toggle_capture(self):
            """Inicia/para captura."""
            if self.video_thread is None or not self.video_thread.running:
                self._start_capture()
            else:
                self._stop_capture()
        
        def _start_capture(self):
            """Inicia captura."""
            self.video_thread = VideoThread(self.monitor)
            self.video_thread.frame_ready.connect(self._update_frame)
            self.video_thread.result_ready.connect(self._update_result)
            self.video_thread.start()
            
            self.start_btn.setText("Parar")
        
        def _stop_capture(self):
            """Para captura."""
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread.wait()
                self.video_thread = None
            
            self.start_btn.setText("Iniciar")
            self.video_label.clear()
        
        def _update_frame(self, frame: np.ndarray):
            """Atualiza frame de vídeo."""
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            self.video_label.setPixmap(
                pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
            )
        
        def _update_result(self, result: HeartRateResult):
            """Atualiza resultado."""
            if result.heart_rate > 0:
                self.hr_label.setText(f"{result.heart_rate:.0f} BPM")
                
                if result.confidence > 0.7:
                    self.hr_label.setStyleSheet("color: #00ff00;")
                elif result.confidence > 0.5:
                    self.hr_label.setStyleSheet("color: #ffff00;")
                else:
                    self.hr_label.setStyleSheet("color: #ff8800;")
            else:
                self.hr_label.setText("-- BPM")
                self.hr_label.setStyleSheet("color: #888888;")
            
            self.confidence_label.setText(f"Confiança: {result.confidence:.0%}")
            self.quality_bar.setValue(int(result.signal_quality * 100))
        
        def _on_mode_changed(self, index: int):
            """Callback de mudança de modo."""
            if index == 0:
                self.monitor.mode = CaptureMode.FINGER
            else:
                self.monitor.mode = CaptureMode.FACE
            
            self.monitor.signal_buffer.clear()
            self.monitor.current_hr = 0.0
        
        def closeEvent(self, event):
            """Evento de fechar janela."""
            self._stop_capture()
            event.accept()


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor de Frequência Cardíaca')
    parser.add_argument('--cli', action='store_true', help='Usar interface CLI')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Índice da câmera')
    
    args = parser.parse_args()
    
    if args.cli or not HAS_PYQT:
        run_cli_app(camera=args.camera)
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
