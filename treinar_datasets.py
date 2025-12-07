import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
import pickle

# Carrega os dados do CSV
data = pd.read_csv('landmarks.csv')

# Separa as features (coordenadas) dos labels (classes)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Codifica os labels (nomes de gestos) para números (0, 1, 2...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Constrói o modelo da rede neural (MLP - Multi-Layer Perceptron)
num_classes = len(label_encoder.classes_)
input_dim = X_train.shape[1] # 42 = 21 landmarks * 2 coordenadas (x,y)

model = Sequential([
    InputLayer(input_shape=(input_dim,)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Treina o modelo
print("\nIniciando o treinamento...")
model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.1)

# Avalia o modelo
print("\nAvaliando o modelo...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Salva o modelo e o label encoder
print("\nSalvando modelo e label encoder...")
model.save('landmark_model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Treinamento concluído com sucesso!")