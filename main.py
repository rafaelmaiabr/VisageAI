import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import mss
import mss.tools

# Dicionário de tradução para as emoções
TRADUCOES = {
    'angry': 'Raiva',
    'disgust': 'Nojo',
    'fear': 'Medo',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Surpreso',
    'neutral': 'Neutro'
}

# Inicialização do MediaPipe para detecção facial
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def capture_screen():
    """
    Captura a tela do computador
    """
    with mss.mss() as sct:
        # Captura a tela principal
        monitor = sct.monitors[1]  # monitor principal
        screenshot = sct.grab(monitor)
        # Converte para numpy array
        frame = np.array(screenshot)
        # Converte de BGRA para BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

def analyze_emotion(frame):
    """
    Analisa a emoção em um frame usando DeepFace
    Retorna a emoção predominante e o dicionário completo de emoções
    """
    try:
        # Análise de emoção com DeepFace
        result = DeepFace.analyze(frame, 
                                actions=['emotion'],
                                enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        # Traduz a emoção predominante
        dominant = TRADUCOES.get(result['dominant_emotion'], result['dominant_emotion'])
        
        # Traduz todas as emoções no dicionário
        emotions_pt = {TRADUCOES.get(k, k): v for k, v in result['emotion'].items()}
            
        return dominant, emotions_pt
    except Exception as e:
        print(f"Erro na análise de emoção: {str(e)}")
        return None, None

def main():
    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    
    # Verifica se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao abrir a webcam!")
        return

    print("Pressione 'q' para sair")
    print("Pressione 'c' para alternar entre webcam e captura de tela")

    # Tempo para controlar a frequência de análise de emoção
    last_emotion_time = time.time()
    emotion_interval = 1.0  # Intervalo em segundos
    current_emotion = "Neutro"
    emotion_scores = {}
    
    # Controle do modo de captura
    using_webcam = True

    while True:
        try:
            if using_webcam:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao capturar frame da webcam!")
                    break
            else:
                frame = capture_screen()

            # Converte o frame para RGB (MediaPipe usa RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta faces no frame
            results = face_detection.process(frame_rgb)

            # Redimensiona o frame se for captura de tela (para melhor visualização)
            if not using_webcam:
                height, width = frame.shape[:2]
                new_width = 1280  # largura fixa
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))

            # Desenha as detecções e análise de emoção
            if results.detections:
                for detection in results.detections:
                    # Desenha o retângulo da detecção facial
                    mp_drawing.draw_detection(frame, detection)

                    # Analisa emoção a cada intervalo
                    if time.time() - last_emotion_time > emotion_interval:
                        emotion, scores = analyze_emotion(frame)
                        if emotion:
                            current_emotion = emotion
                            emotion_scores = scores
                        last_emotion_time = time.time()

            # Adiciona texto com a emoção atual
            cv2.putText(frame, f"Emoção: {current_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostra o modo atual
            modo_texto = "Webcam" if using_webcam else "Captura de Tela"
            cv2.putText(frame, f"Modo: {modo_texto}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Se tiver scores de emoção, mostra eles
            if emotion_scores:
                y_pos = 60
                for emotion, score in emotion_scores.items():
                    text = f"{emotion}: {score:.2f}"
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20

            # Mostra o frame
            cv2.imshow('Analise de Sentimento em Tempo Real', frame)

            # Verifica teclas pressionadas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Sair
                break
            elif key == ord('c'):  # Alternar modo de captura
                using_webcam = not using_webcam
                print(f"Alternando para modo: {'Webcam' if using_webcam else 'Captura de Tela'}")

        except Exception as e:
            print(f"Erro: {str(e)}")
            break

    # Libera os recursos
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()