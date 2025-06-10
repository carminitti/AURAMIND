import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

def extrair_caracteristicas(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        return np.hstack([mfccs, chroma, zcr, rms])
    except Exception as e:
        print(f"Erro ao processar {audio_path}: {e}")
        return None

# Caminho base
base_path = "dados"

X = []
y = []

# Percorre subpastas (ex: feliz/, triste/, raiva/)
for rotulo in os.listdir(base_path):
    pasta = os.path.join(base_path, rotulo)
    if os.path.isdir(pasta):
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".wav"):
                caminho_audio = os.path.join(pasta, arquivo)
                print(f"Processando: {caminho_audio}")
                caracteristicas = extrair_caracteristicas(caminho_audio)
                if caracteristicas is not None:
                    X.append(caracteristicas)
                    y.append(rotulo)

# Verifica se há dados suficientes
if len(set(y)) < 2 or len(X) < 3:
    print("Dados insuficientes para treinar um modelo confiável.")
    exit()

# Codifica os rótulos
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Divide os dados corretamente
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# Treina o modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Avaliação real
y_pred = modelo.predict(X_test)
print("\n📊 Avaliação em dados não vistos:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Salva modelo e encoder
joblib.dump(modelo, r"Z:\TCC\emocao_audio_api\modelo_audio.pkl")
joblib.dump(encoder, r"Z:\TCC\emocao_audio_api\encoder.pkl")

print("\n Modelo treinado e salvo com sucesso.")
