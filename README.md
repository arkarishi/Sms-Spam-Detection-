# SMS/Email Spam Classifier

## 📌 Project Overview
This project is a **Spam Classifier** that detects whether an SMS or email message is **Spam 🚨** or **Not Spam ✅** using **Machine Learning**. It is built using **Python, Scikit-Learn, NLTK, and Streamlit** for an interactive UI.

## 🏗 Features
- Preprocesses text messages (removes stopwords, punctuations, and applies stemming)
- Converts text data into numerical format using **TF-IDF Vectorization**
- Uses **Multinomial Naïve Bayes (MultinomialNB)** as the machine learning model
- Interactive web app using **Streamlit**
- Handles missing model/vectorizer files and alerts users if they need to be trained first

## 📂 Folder Structure
```
📦 SMS-Email-Spam-Classifier
 ┣ 📜 app.py              # Streamlit Web App
 ┣ 📜 model.pkl           # Trained ML Model (Naïve Bayes)
 ┣ 📜 vectorizer.pkl      # TF-IDF Vectorizer
 ┣ 📜 spam.csv           # Dataset (Spam/Ham messages)
 ┣ 📜 train_model.py      # Script to train and save the model
 ┣ 📜 README.md           # Project Documentation
```

## ⚙️ Setup & Installation

### **1️⃣ Install Dependencies**
Make sure you have **Python 3.10+** installed, then install the required libraries:
```bash
pip install streamlit pandas scikit-learn nltk pickle-mixin
```

### **2️⃣ Download NLTK Stopwords & Tokenizer**
Run this inside a Python script or Jupyter Notebook:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### **3️⃣ Run the Streamlit App**
Launch the web app using:
```bash
streamlit run app.py
```

## 🚀 Training the Model
If `model.pkl` and `vectorizer.pkl` are missing, train the model using:
```bash
python train_model.py
```
This will process the dataset and save the trained model and vectorizer.

## 🛠 How It Works
1. **User enters a message** → The text is preprocessed and transformed.
2. **TF-IDF Vectorization** → Converts text to numerical format.
3. **Prediction** → The trained Naïve Bayes model classifies the message as **Spam** or **Not Spam**.
4. **Result Displayed** → The output is shown in the Streamlit UI.

## 📊 Model Performance (Example)
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98%    |
| Precision | 96%    |
| Recall    | 92%    |

## 🔗 Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NLTK Documentation](https://www.nltk.org/)

---

### **💡 Future Enhancements**
- Add support for **deep learning models** (LSTMs, Transformers)
- Improve UI with **better visualization of results**
- Deploy the model as a **web API using FastAPI or Flask**

📌 **Built with ❤️ by Arka Provo Sarkar**
