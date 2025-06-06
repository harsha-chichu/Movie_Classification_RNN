
# 🎬 Movie Review Classification using RNN

This project demonstrates a **Recurrent Neural Network (RNN)** model for binary classification of **movie reviews** (positive or negative sentiment). It uses the **IMDB dataset** and employs an embedding layer followed by a SimpleRNN layer to perform sentiment analysis.

---

## 🚀 Features

- Binary sentiment classification (positive/negative)
- Uses Keras' built-in IMDB dataset
- Tokenization and sequence padding
- Simple and interpretable RNN architecture
- Training visualization with accuracy/loss plots

---

## 📁 Project Structure

```
Movie_Classification_RNN/
│
├── movie_classification_rnn.ipynb  # Jupyter Notebook with the complete workflow
├── requirements.txt                # Required Python packages
└── README.md                       # Project documentation
```

---

## 📚 How It Works

1. **Dataset**: IMDB dataset from Keras with 50,000 labeled movie reviews.
2. **Preprocessing**:
   - Word indices limited to the top 10,000 most frequent words
   - Padding sequences to ensure equal input length
3. **Model Architecture**:
   - Embedding layer to map word indices to dense vectors
   - SimpleRNN layer to learn temporal patterns
   - Dense output layer with sigmoid activation
4. **Training**:
   - Binary crossentropy loss and Adam optimizer
   - Trained over multiple epochs with validation

---

## 🧪 Example Output

Model will output a probability between 0 and 1:
- **> 0.5** → Positive review
- **< 0.5** → Negative review

---

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/harsha-chichu/Movie_Classification_RNN.git
   cd Movie_Classification_RNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   Open `movie_classification_rnn.ipynb` using Jupyter or VS Code and run the cells step-by-step.

---

## 📦 Requirements

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- matplotlib

You can install them using:
```bash
pip install tensorflow keras numpy matplotlib
```

---

## ✍️ Author

**Harsha Vardhan**  
🔗 [GitHub](https://github.com/harsha-chichu)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🌟 Star this repo

If you found this project helpful, consider giving it a ⭐️ to show your support!
