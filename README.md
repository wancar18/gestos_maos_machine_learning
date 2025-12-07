# gestos_maos_machine_learning
Tranbalho RA professor Yhuri

# Reconhecimento de Gestos com Mãos (Machine Learning)

Este projeto utiliza **MediaPipe** para capturar pontos de referência (landmarks) das mãos e **TensorFlow/Keras** para classificar gestos personalizados em tempo real.

---

## 📋 Pré-requisitos

Devido a restrições de compatibilidade da biblioteca `mediapipe`, é **obrigatório** utilizar uma versão do Python compatível:

* **Python 3.8 até 3.11** (Recomendado: **Python 3.11**)
* *Nota: O Python 3.13 ainda NÃO é suportado pelo MediaPipe (especialmente em macOS).*

---

## 🚀 Instalação e Configuração

Siga os passos abaixo de acordo com o seu sistema operacional.

### 1. Clonar o repositório
```bash
git clone [https://github.com/wancar18/gestos_maos_machine_learning.git](https://github.com/wancar18/gestos_maos_machine_learning.git)
cd gestos_maos_machine_learning

# Se tiver o Python 3.11 instalado:
py -3.11 -m venv .venv
# OU apenas:
python -m venv .venv

Windows
.venv\Scripts\activate

MacOS
source .venv/bin/activate

