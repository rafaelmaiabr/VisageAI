import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time
import mss
import mss.tools

# Variáveis globais
selected_region = None  # Armazena a região selecionada
active_monitor = 1      # Monitor ativo para captura

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

def select_screen_region():
    """
    Permite ao usuário selecionar uma região da tela para captura
    Retorna: (x, y, width, height) da região selecionada ou None se nenhuma região for selecionada
    """
    global active_monitor
    # Captura inicial da tela completa
    with mss.mss() as sct:
        monitor = sct.monitors[active_monitor]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Redimensiona a imagem para uma visualização melhor
        height, width = img.shape[:2]
        new_width = 1280
        new_height = int(height * (new_width / width))
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # Fator de escala para converter coordenadas de volta ao tamanho original
        scale_x = width / new_width
        scale_y = height / new_height
        
        # Cria janela e define callback para seleção
        window_name = "Selecione a região (arraste o mouse) - Pressione ENTER para confirmar, ESC para cancelar"
        cv2.namedWindow(window_name)
        roi = cv2.selectROI(window_name, img_resized, False)
        cv2.destroyWindow(window_name)
        
        # Verifica se uma região foi selecionada (width e height > 0)
        if roi[2] > 0 and roi[3] > 0:
            # Converte as coordenadas de volta para o tamanho original da tela
            x = int(roi[0] * scale_x)
            y = int(roi[1] * scale_y)
            w = int(roi[2] * scale_x)
            h = int(roi[3] * scale_y)
            
            return {"top": y, "left": x, "width": w, "height": h}
        return None

def get_active_monitor():
    """
    Obtém informações sobre o monitor ativo (onde a janela está sendo exibida)
    """
    with mss.mss() as sct:
        # Lista todos os monitores
        for i, monitor in enumerate(sct.monitors[1:], 1):  # Skip the "all-in-one" monitor (index 0)
            print(f"Monitor {i}: {monitor}")
        
        # Por padrão, usa o monitor principal
        primary_monitor = next((i for i, m in enumerate(sct.monitors[1:], 1) if m.get('primary')), 1)
        return primary_monitor

def capture_screen():
    """
    Captura a tela do computador.
    Se houver uma região selecionada, captura apenas essa região.
    Caso contrário, captura a tela inteira.
    """
    global selected_region, active_monitor
    
    with mss.mss() as sct:
        if selected_region:
            # Adiciona o offset do monitor à região selecionada
            monitor_info = sct.monitors[active_monitor]
            region = {
                "top": selected_region["top"] + monitor_info["top"],
                "left": selected_region["left"] + monitor_info["left"],
                "width": selected_region["width"],
                "height": selected_region["height"]
            }
            screenshot = sct.grab(region)
        else:
            # Captura o monitor ativo
            screenshot = sct.grab(sct.monitors[active_monitor])
            
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
    global selected_region, active_monitor  # Declarando acesso às variáveis globais
    
    # Determina o monitor ativo
    active_monitor = get_active_monitor()
    print(f"\nUsando monitor {active_monitor} para captura")
    print("Para alternar entre monitores, use as teclas numéricas (1, 2, ...)")
    
    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    
    # Verifica se a webcam foi aberta corretamente
    using_webcam = True
    if not cap.isOpened():
        print("Erro ao abrir a webcam! Iniciando com captura de tela...")
        using_webcam = False
    else:
        # Tenta capturar um frame para verificar se a webcam está funcionando
        ret, _ = cap.read()
        if not ret:
            print("Erro ao capturar frame da webcam! Iniciando com captura de tela...")
            cap.release()
            using_webcam = False

    print("\nComandos disponíveis:")
    print("- 'q': sair")
    print("- 'c': alternar entre webcam e captura de tela")
    print("- 'r': selecionar uma região específica (modo captura)")
    print("- 'f': voltar à captura em tela cheia (modo captura)")
    print("- '1-9': selecionar monitor para captura (modo captura)")

    # Tempo para controlar a frequência de análise de emoção
    last_emotion_time = time.time()
    emotion_interval = 1.0  # Intervalo em segundos
    current_emotion = "Neutro"
    emotion_scores = {}

    while True:
        try:
            if using_webcam:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao capturar frame da webcam! Alternando para captura de tela...")
                    using_webcam = False
                    continue
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
            modo_texto = "Webcam" if using_webcam else f"Captura de Tela (Monitor {active_monitor})"
            modo_texto += " (Região Selecionada)" if selected_region else ""
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
                if not using_webcam:
                    print("\nModo: Captura de Tela")
                    print("Comandos disponíveis:")
                    print("- 'r': selecionar uma região específica")
                    print("- 'f': voltar para captura em tela cheia")
                    print("- '1-9': selecionar monitor diferente")
                else:
                    # Limpa a região selecionada ao voltar para webcam
                    selected_region = None
                    print("\nModo: Webcam")
            elif key == ord('r') and not using_webcam:  # Selecionar nova região
                print("\nSelecione a região da tela para captura...")
                print("1. Clique e arraste para selecionar a área")
                print("2. Pressione ENTER para confirmar")
                print("3. Pressione ESC para cancelar\n")
                
                nova_regiao = select_screen_region()
                if nova_regiao:
                    selected_region = nova_regiao
                    print("Nova região selecionada com sucesso!")
                else:
                    print("Seleção cancelada, mantendo configuração anterior.")
            elif key == ord('f') and not using_webcam:  # Voltar para tela cheia
                selected_region = None
                print("\nVoltando para captura em tela cheia.")
            # Alternar entre monitores usando teclas numéricas
            elif not using_webcam and ord('1') <= key <= ord('9'):
                monitor_num = key - ord('0')  # Converte a tecla para número
                with mss.mss() as sct:
                    if monitor_num <= len(sct.monitors[1:]):  # Ignora monitor 0 (combinado)
                        active_monitor = monitor_num
                        selected_region = None  # Reset região ao trocar monitor
                        print(f"\nAlterado para monitor {active_monitor}")
                    else:
                        print(f"\nMonitor {monitor_num} não disponível")

        except Exception as e:
            print(f"Erro: {str(e)}")
            break

    # Libera os recursos
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()