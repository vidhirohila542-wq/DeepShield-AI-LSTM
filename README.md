# 🧠 DeepShield AI

## Deep Learning Based Suspicious Message Detector

DeepShield AI is a beginner-friendly **Deep Learning + Natural Language Processing (NLP)** project that uses an **LSTM (Long Short-Term Memory) neural network** to classify text messages as:

- ✅ Safe
- 🚨 Suspicious / Scam

The project demonstrates how a neural network can learn patterns in text and use them to make a classification.

---

## 🎯 Objective

The goal is to build a Deep Learning model that can identify suspicious messages and help demonstrate how AI can be applied to digital safety.

---

## 🧠 Deep Learning Architecture

The project uses:

1. **Tokenizer** — converts words into numerical sequences.
2. **Padding** — makes input sequences the same length.
3. **Embedding Layer** — learns numerical representations of words.
4. **LSTM Layer** — learns patterns and word relationships in sequences.
5. **Dropout** — helps reduce overfitting.
6. **Dense Layers** — perform the final classification.
7. **Sigmoid Output** — produces a probability for the scam class.

### Model Flow

```text
Text Message
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Padding
     ↓
Embedding Layer
     ↓
LSTM Layer
     ↓
Dropout
     ↓
Dense Layer
     ↓
Sigmoid Output
     ↓
SAFE / SCAM
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- NLP
- LSTM Neural Network

---

## 📂 Project Structure

```text
DeepShield_AI/
│
├── deepshield_lstm.py
├── messages_dataset.csv
├── requirements.txt
├── README.md
├── sample_predictions.txt
└── .gitignore
```

---

## 🚀 How to Run

### 1. Install Python

Install Python 3 on your computer.

### 2. Open the project in VS Code

Open the `DeepShield_AI` folder.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python deepshield_lstm.py
```

The model will train on the included dataset, show its test accuracy and classification report, save the trained neural network, and test sample messages.

---

## 💬 Example

### Input

```text
Congratulations you won a prize click the link to claim now
```

Possible output:

```text
Prediction: SUSPICIOUS / SCAM
```

### Another Input

```text
Please send me the notes from today's lecture
```

Possible output:

```text
Prediction: SAFE
```

---

## 📊 Dataset

The repository contains a small **synthetic educational dataset** with safe and suspicious messages.

The dataset is intended for learning and demonstration. It is not large enough to represent all real-world scams.

---

## 🔮 Future Improvements

- Train on a much larger ethically sourced dataset.
- Add Hindi and Hinglish message detection.
- Detect suspicious URLs separately.
- Add a real-time web interface.
- Compare LSTM with GRU and Transformer models.
- Add explainable AI.
- Deploy the model as an API or web application.

---

## 👩‍💻 Author

**Vidhi Rohila**  
B.Tech CSE - AI/ML

---

## ⚠️ Disclaimer

This is an educational Deep Learning project. Its predictions should not be treated as proof that a message is fraudulent. Suspicious messages should always be verified through trusted official channels.
