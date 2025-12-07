import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Mapeamento de teclas para nomes de gestos
# Adicione ou altere conforme suas pastas de gestos
GESTURE_MAP = {
    'a': 'aberta',
    'f': 'fechada',
    'l': 'like',
    'o': 'ok',
    'r': 'arminha',
    'p': 'paz_e_amor',

    # Adicione mais teclas e nomes de gestos aqui...
}

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Abre a webcam
cap = cv2.VideoCapture(0)

# Nome do arquivo CSV para salvar os dados
csv_file = 'landmarks.csv'
file_exists = os.path.isfile(csv_file)

print("Pressione a tecla correspondente ao gesto para capturar os dados.")
print("Pressione 'q' para sair.")
print("Mapeamento de Teclas:", GESTURE_MAP)


with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    # Escreve o cabeçalho apenas se o arquivo for novo
    if not file_exists:
        header = ['class']
        for i in range(21):
            header += [f'x{i}', f'y{i}'] # Apenas coordenadas x, y
        writer.writerow(header)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os landmarks na tela para feedback visual
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                key = cv2.waitKey(10) & 0xFF

                if key != 255 and chr(key) in GESTURE_MAP:
                    gesture_class = GESTURE_MAP[chr(key)]
                    print(f"Capturando gesto: {gesture_class}")

                    # --- Normalização dos Landmarks ---
                    landmarks = hand_landmarks.landmark
                    # Pega as coordenadas do pulso (ponto 0) para usar como referência
                    wrist_coords = np.array([landmarks[0].x, landmarks[0].y])
                    
                    # Extrai e normaliza todas as coordenadas relativas ao pulso
                    normalized_landmarks = []
                    for lm in landmarks:
                        normalized_landmarks.append(lm.x - wrist_coords[0])
                        normalized_landmarks.append(lm.y - wrist_coords[1])
                    
                    # Salva a linha no CSV
                    row = [gesture_class] + normalized_landmarks
                    writer.writerow(row)

        cv2.imshow('Coleta de Landmarks', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Dados salvos em {csv_file}")

#{'a': 'aberta', 'f': 'fechada', 'l': 'like', 'o': 'ok', 'p': 'paz_e_amor'}