"""
=============================================================================
Módulo de Detecção Facial e Extração de ROI para rPPG
=============================================================================
Este módulo implementa a detecção facial e extração de regiões de interesse
(ROI) para fotopletismografia remota (rPPG).

A extração de ROI é crucial para rPPG pois a qualidade do sinal depende
diretamente da região da pele analisada.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class ROIType(Enum):
    """Tipos de região de interesse para rPPG."""
    FOREHEAD = "forehead"       # Testa (melhor SNR)
    LEFT_CHEEK = "left_cheek"   # Bochecha esquerda
    RIGHT_CHEEK = "right_cheek" # Bochecha direita
    FULL_FACE = "full_face"     # Face completa
    COMBINED = "combined"        # Combinação de regiões


@dataclass
class FaceDetection:
    """
    Classe de dados para armazenar resultado de detecção facial.
    
    Atributos:
        bbox: Bounding box da face (x, y, largura, altura)
        landmarks: Pontos de referência facial (opcional)
        confidence: Confiança da detecção
    """
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[np.ndarray] = None
    confidence: float = 1.0


class FaceDetector:
    """
    Detector de faces usando diferentes backends.
    
    Suporta múltiplos métodos de detecção:
    - Haar Cascades (OpenCV clássico, rápido)
    - DNN Face Detector (OpenCV DNN, mais preciso)
    - MediaPipe (moderno, com landmarks)
    
    Exemplo de Uso:
        >>> detector = FaceDetector(method='haar')
        >>> face = detector.detect(frame)
        >>> if face:
        ...     print(f"Face detectada em: {face.bbox}")
    """
    
    def __init__(self, method: str = 'haar'):
        """
        Inicializa o detector de faces.
        
        Args:
            method: Método de detecção ('haar', 'dnn', 'mediapipe')
        """
        self.method = method
        self._detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Inicializa o detector baseado no método escolhido."""
        if self.method == 'haar':
            # Carregar classificador Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._detector = cv2.CascadeClassifier(cascade_path)
            
        elif self.method == 'dnn':
            # Carregar modelo DNN pré-treinado
            # Nota: Requer arquivos do modelo
            model_file = "models/opencv_face_detector_uint8.pb"
            config_file = "models/opencv_face_detector.pbtxt"
            try:
                self._detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            except Exception:
                # Fallback para Haar se modelo não disponível
                print("Modelo DNN não encontrado, usando Haar Cascade")
                self.method = 'haar'
                self._initialize_detector()
                
        elif self.method == 'mediapipe':
            try:
                import mediapipe as mp
                self._mp_face_detection = mp.solutions.face_detection
                self._mp_face_mesh = mp.solutions.face_mesh
                self._detector = self._mp_face_detection.FaceDetection(
                    model_selection=1,  # 1 = full range model
                    min_detection_confidence=0.5
                )
                self._face_mesh = self._mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                )
            except ImportError:
                print("MediaPipe não instalado, usando Haar Cascade")
                self.method = 'haar'
                self._initialize_detector()
    
    def detect(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Detecta face no frame.
        
        Args:
            frame: Frame BGR do OpenCV
        
        Returns:
            FaceDetection com informações da face, ou None se não detectada
        """
        if self.method == 'haar':
            return self._detect_haar(frame)
        elif self.method == 'dnn':
            return self._detect_dnn(frame)
        elif self.method == 'mediapipe':
            return self._detect_mediapipe(frame)
        return None
    
    def _detect_haar(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """Detecção usando Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Pegar a maior face detectada
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        return FaceDetection(
            bbox=(int(x), int(y), int(w), int(h)),
            confidence=1.0
        )
    
    def _detect_dnn(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """Detecção usando DNN."""
        h, w = frame.shape[:2]
        
        # Preparar blob para a rede
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False
        )
        
        self._detector.setInput(blob)
        detections = self._detector.forward()
        
        # Encontrar detecção com maior confiança
        best_detection = None
        best_confidence = 0.5  # Threshold mínimo
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                best_detection = (x1, y1, x2 - x1, y2 - y1)
                best_confidence = confidence
        
        if best_detection:
            return FaceDetection(
                bbox=best_detection,
                confidence=float(best_confidence)
            )
        return None
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """Detecção usando MediaPipe."""
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detectar face
        results = self._detector.process(rgb_frame)
        
        if not results.detections:
            return None
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Converter coordenadas relativas para absolutas
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Obter landmarks faciais
        mesh_results = self._face_mesh.process(rgb_frame)
        landmarks = None
        
        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            landmarks = np.array([
                [lm.x * w, lm.y * h, lm.z]
                for lm in face_landmarks.landmark
            ])
        
        return FaceDetection(
            bbox=(x, y, width, height),
            landmarks=landmarks,
            confidence=detection.score[0]
        )


class ROIExtractor:
    """
    Extrator de Regiões de Interesse (ROI) para rPPG.
    
    Extrai regiões específicas da face que são mais adequadas
    para análise de fotopletismografia remota.
    
    Regiões recomendadas para rPPG:
    - Testa: Maior área de pele, menos movimento, bom SNR
    - Bochechas: Boa vascularização, sinal forte
    
    Exemplo de Uso:
        >>> extractor = ROIExtractor()
        >>> roi_mask = extractor.get_roi_mask(frame, face_detection, ROIType.FOREHEAD)
    """
    
    # Índices dos landmarks do MediaPipe para diferentes regiões
    # Referência: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    FOREHEAD_LANDMARKS = [10, 67, 109, 108, 107, 55, 8, 285, 336, 337, 338, 297]
    LEFT_CHEEK_LANDMARKS = [116, 117, 118, 119, 100, 126, 209, 49, 129]
    RIGHT_CHEEK_LANDMARKS = [345, 346, 347, 348, 329, 355, 429, 279, 358]
    
    def __init__(self):
        """Inicializa o extrator de ROI."""
        pass
    
    def get_roi_mask(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        roi_type: ROIType = ROIType.FOREHEAD
    ) -> np.ndarray:
        """
        Obtém máscara binária da região de interesse.
        
        Args:
            frame: Frame BGR
            face: Detecção facial com bbox e landmarks
            roi_type: Tipo de ROI desejada
        
        Returns:
            Máscara binária (0/255) com a ROI
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if face.landmarks is not None:
            # Usar landmarks para ROI precisa
            return self._get_roi_from_landmarks(frame, face.landmarks, roi_type)
        else:
            # Usar bounding box para ROI aproximada
            return self._get_roi_from_bbox(frame, face.bbox, roi_type)
    
    def _get_roi_from_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        roi_type: ROIType
    ) -> np.ndarray:
        """Extrai ROI usando landmarks faciais."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if roi_type == ROIType.FOREHEAD:
            indices = self.FOREHEAD_LANDMARKS
        elif roi_type == ROIType.LEFT_CHEEK:
            indices = self.LEFT_CHEEK_LANDMARKS
        elif roi_type == ROIType.RIGHT_CHEEK:
            indices = self.RIGHT_CHEEK_LANDMARKS
        elif roi_type == ROIType.COMBINED:
            # Combinar múltiplas regiões
            mask1 = self._get_roi_from_landmarks(frame, landmarks, ROIType.FOREHEAD)
            mask2 = self._get_roi_from_landmarks(frame, landmarks, ROIType.LEFT_CHEEK)
            mask3 = self._get_roi_from_landmarks(frame, landmarks, ROIType.RIGHT_CHEEK)
            return cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        else:
            # Full face - usar convex hull de todos os landmarks
            points = landmarks[:, :2].astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
            return mask
        
        # Criar polígono da ROI
        if len(landmarks) > max(indices):
            points = landmarks[indices, :2].astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
        
        return mask
    
    def _get_roi_from_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        roi_type: ROIType
    ) -> np.ndarray:
        """
        Extrai ROI usando apenas bounding box.
        
        Usa proporções aproximadas da face para definir regiões.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y, bw, bh = bbox
        
        if roi_type == ROIType.FOREHEAD:
            # Testa: região superior da face
            roi_x = x + int(bw * 0.25)
            roi_y = y + int(bh * 0.05)
            roi_w = int(bw * 0.5)
            roi_h = int(bh * 0.2)
            
        elif roi_type == ROIType.LEFT_CHEEK:
            # Bochecha esquerda
            roi_x = x + int(bw * 0.05)
            roi_y = y + int(bh * 0.4)
            roi_w = int(bw * 0.25)
            roi_h = int(bh * 0.25)
            
        elif roi_type == ROIType.RIGHT_CHEEK:
            # Bochecha direita
            roi_x = x + int(bw * 0.7)
            roi_y = y + int(bh * 0.4)
            roi_w = int(bw * 0.25)
            roi_h = int(bh * 0.25)
            
        elif roi_type == ROIType.FULL_FACE:
            # Face completa (com margem)
            margin = 0.1
            roi_x = x + int(bw * margin)
            roi_y = y + int(bh * margin)
            roi_w = int(bw * (1 - 2 * margin))
            roi_h = int(bh * (1 - 2 * margin))
            
        elif roi_type == ROIType.COMBINED:
            # Combinação de testa e bochechas
            mask1 = self._get_roi_from_bbox(frame, bbox, ROIType.FOREHEAD)
            mask2 = self._get_roi_from_bbox(frame, bbox, ROIType.LEFT_CHEEK)
            mask3 = self._get_roi_from_bbox(frame, bbox, ROIType.RIGHT_CHEEK)
            return cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        # Garantir que ROI está dentro dos limites
        roi_x = max(0, min(roi_x, w - 1))
        roi_y = max(0, min(roi_y, h - 1))
        roi_w = min(roi_w, w - roi_x)
        roi_h = min(roi_h, h - roi_y)
        
        # Criar máscara retangular com bordas suavizadas
        mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
        
        # Aplicar blur para suavizar bordas
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def extract_roi_values(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Extrai valores médios dos canais RGB na ROI.
        
        Args:
            frame: Frame BGR
            mask: Máscara binária da ROI
        
        Returns:
            Dicionário com valores médios dos canais
        """
        # Converter máscara para binária
        binary_mask = mask > 128
        
        if not np.any(binary_mask):
            return {'red': 0, 'green': 0, 'blue': 0}
        
        # Extrair valores dos canais
        blue = frame[:, :, 0][binary_mask].mean()
        green = frame[:, :, 1][binary_mask].mean()
        red = frame[:, :, 2][binary_mask].mean()
        
        return {
            'blue': float(blue),
            'green': float(green),
            'red': float(red)
        }


class SkinSegmenter:
    """
    Segmentador de pele para melhorar extração de ROI.
    
    Usa detecção de cor de pele para refinar a região de análise,
    excluindo áreas que não são pele (olhos, cabelo, fundo).
    """
    
    # Limites de cor de pele no espaço YCrCb
    YCRCB_MIN = np.array([0, 133, 77], dtype=np.uint8)
    YCRCB_MAX = np.array([255, 173, 127], dtype=np.uint8)
    
    # Limites no espaço HSV
    HSV_MIN = np.array([0, 20, 70], dtype=np.uint8)
    HSV_MAX = np.array([20, 255, 255], dtype=np.uint8)
    
    def __init__(self, method: str = 'ycrcb'):
        """
        Inicializa o segmentador de pele.
        
        Args:
            method: Método de segmentação ('ycrcb', 'hsv', 'combined')
        """
        self.method = method
    
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Segmenta regiões de pele no frame.
        
        Args:
            frame: Frame BGR
        
        Returns:
            Máscara binária das regiões de pele
        """
        if self.method == 'ycrcb':
            return self._segment_ycrcb(frame)
        elif self.method == 'hsv':
            return self._segment_hsv(frame)
        else:
            # Combinar métodos
            mask1 = self._segment_ycrcb(frame)
            mask2 = self._segment_hsv(frame)
            return cv2.bitwise_and(mask1, mask2)
    
    def _segment_ycrcb(self, frame: np.ndarray) -> np.ndarray:
        """Segmentação usando espaço de cor YCrCb."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.YCRCB_MIN, self.YCRCB_MAX)
        
        # Operações morfológicas para limpar a máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _segment_hsv(self, frame: np.ndarray) -> np.ndarray:
        """Segmentação usando espaço de cor HSV."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Dois ranges para capturar tons de pele
        mask1 = cv2.inRange(hsv, self.HSV_MIN, self.HSV_MAX)
        mask2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Operações morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask


class FingerROIExtractor:
    """
    Extrator de ROI para modo dedo (finger PPG).
    
    Quando o dedo é colocado sobre a câmera com flash ligado,
    a luz atravessa o tecido e a variação é detectada.
    """
    
    def __init__(self, roi_percentage: float = 0.6):
        """
        Inicializa o extrator.
        
        Args:
            roi_percentage: Percentual central da imagem a usar
        """
        self.roi_percentage = roi_percentage
    
    def extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrai ROI central para modo dedo.
        
        Args:
            frame: Frame da câmera
        
        Returns:
            Tuple (frame_roi, máscara)
        """
        h, w = frame.shape[:2]
        
        # Calcular região central
        margin_x = int(w * (1 - self.roi_percentage) / 2)
        margin_y = int(h * (1 - self.roi_percentage) / 2)
        
        x1 = margin_x
        y1 = margin_y
        x2 = w - margin_x
        y2 = h - margin_y
        
        # Criar máscara
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Extrair ROI
        roi = frame[y1:y2, x1:x2]
        
        return roi, mask
    
    def get_mean_values(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Obtém valores médios dos canais RGB.
        
        Args:
            frame: Frame da câmera
        
        Returns:
            Dicionário com valores médios
        """
        roi, _ = self.extract_roi(frame)
        
        return {
            'blue': float(roi[:, :, 0].mean()),
            'green': float(roi[:, :, 1].mean()),
            'red': float(roi[:, :, 2].mean())
        }
    
    def is_finger_present(self, frame: np.ndarray, threshold: float = 50) -> bool:
        """
        Verifica se há dedo sobre a câmera.
        
        Quando o dedo está presente:
        - Canal vermelho dominante (luz atravessa tecido)
        - Alta intensidade geral (flash ligado)
        - Baixa variância espacial (área uniforme)
        
        Args:
            frame: Frame da câmera
            threshold: Limiar para detecção
        
        Returns:
            True se dedo detectado
        """
        roi, _ = self.extract_roi(frame)
        
        # Verificar se canal vermelho é dominante
        red_mean = roi[:, :, 2].mean()
        green_mean = roi[:, :, 1].mean()
        blue_mean = roi[:, :, 0].mean()
        
        # Critérios para detecção de dedo
        is_red_dominant = red_mean > green_mean > blue_mean
        is_bright = red_mean > threshold
        is_uniform = np.std(roi[:, :, 2]) < red_mean * 0.5
        
        return is_red_dominant and is_bright and is_uniform


# =============================================================================
# Funções utilitárias
# =============================================================================

def visualize_roi(
    frame: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Visualiza a ROI sobreposta ao frame.
    
    Args:
        frame: Frame BGR original
        mask: Máscara da ROI
        alpha: Transparência da sobreposição
        color: Cor da sobreposição (BGR)
    
    Returns:
        Frame com ROI visualizada
    """
    overlay = frame.copy()
    
    # Criar máscara colorida
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 128] = color
    
    # Combinar
    output = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    return output


def draw_face_detection(
    frame: np.ndarray,
    face: FaceDetection,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Desenha detecção facial no frame.
    
    Args:
        frame: Frame BGR
        face: Detecção facial
        color: Cor do desenho
        thickness: Espessura das linhas
    
    Returns:
        Frame com detecção desenhada
    """
    output = frame.copy()
    x, y, w, h = face.bbox
    
    # Desenhar bounding box
    cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
    
    # Desenhar landmarks se disponíveis
    if face.landmarks is not None:
        for point in face.landmarks[:468]:  # Limitar pontos
            px, py = int(point[0]), int(point[1])
            cv2.circle(output, (px, py), 1, (255, 0, 0), -1)
    
    # Mostrar confiança
    text = f"Conf: {face.confidence:.2f}"
    cv2.putText(output, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)
    
    return output


if __name__ == "__main__":
    # Teste do módulo
    print("Testando módulo de detecção facial e extração de ROI...")
    
    # Criar frame de teste
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Testar detector
    detector = FaceDetector(method='haar')
    face = detector.detect(test_frame)
    print(f"Detecção facial: {face}")
    
    # Testar extrator de ROI para dedo
    finger_extractor = FingerROIExtractor()
    values = finger_extractor.get_mean_values(test_frame)
    print(f"Valores médios (dedo): {values}")
    
    print("Testes concluídos!")
