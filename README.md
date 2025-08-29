# 🚦 Traffic Sign Recognition System  

A deep learning-based project that uses *Convolutional Neural Networks (CNNs)* to classify traffic signs from images. This system is designed to assist in *road safety, driver assistance, and autonomous vehicles*.  

---

## 📌 Project Overview  
Traffic signs play a vital role in regulating road traffic. This system automatically recognizes traffic signs using *image processing + deep learning, trained on the **German Traffic Sign Recognition Benchmark (GTSRB)* dataset.  

---

## 📊 Dataset  
- *Dataset:* GTSRB (German Traffic Sign Recognition Benchmark)  
- *Classes:* 43 traffic sign categories  
- *Training Images:* ~39,000  
- *Testing Images:* ~12,000  

---

## ⚙ Tech Stack  
- *Language:* Python  
- *Libraries:* TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib  
- *Model:* Convolutional Neural Network (CNN)  

---

## 🔬 Methodology  
1. *Data Preprocessing* – resizing, normalization, train/test split  
2. *Model Building* – CNN with Conv, Pooling, Dropout, and Dense layers  
3. *Training & Validation* – accuracy/loss monitored  
4. *Testing & Prediction* – tested on unseen images  

---

## 🚀 Applications  
- Self-driving cars 🛻  
- Driver assistance systems 🚘  
- Intelligent traffic management 🌆  

---
---

## 🛠 Installation & Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/username/Traffic-Sign-Recognition.git
   cd Traffic-Sign-Recognition

   pip install -r requirements.txt
   python Traffic_sign_recognition.py