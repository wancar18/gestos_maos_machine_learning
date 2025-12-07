import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Carrega o modelo treinado e o label encoder
model = tf.keras.models.load_model('landmark_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inicia a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            wrist_coords = np.array([landmarks[0].x, landmarks[0].y])
            
            normalized_landmarks = []
            for lm in landmarks:
                normalized_landmarks.append(lm.x - wrist_coords[0])
                normalized_landmarks.append(lm.y - wrist_coords[1])

            data_point = np.array(normalized_landmarks).reshape(1, -1)

            prediction = model.predict(data_point)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
            confidence = np.max(prediction) * 100

            cv2.putText(frame, f"{predicted_class_name} ({confidence:.2f}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ### NOVO: Exibir lista de gestos conhecidos pelo modelo no canto da tela ###
    # Obtém a altura e largura da imagem para posicionar o texto (opcional, para alinhar à direita)
    h, w, c = frame.shape
    
    cv2.putText(frame, "Gestos Disponiveis:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # label_encoder.classes_ contém a lista ['aberta', 'arminha', 'fechada', etc.]
    for i, gesto in enumerate(label_encoder.classes_):
        y_pos = 90 + (i * 20)
        cv2.putText(frame, f"- {gesto}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # #########################################################################

    cv2.imshow('Detecção com Landmarks', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()