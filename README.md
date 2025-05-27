# 🎬 Movie Classification using Simple RNN

This project implements a sentiment analysis model using a Recurrent Neural Network (RNN) to classify movie reviews from the IMDB dataset as **positive** or **negative**. The model is built using TensorFlow/Keras with an Embedding layer, a SimpleRNN layer, and a Dense output layer.

---

## 📁 Project Structure

├── simplernn.ipynb # Jupyter notebook for training and evaluation
├── prediction.ipynb # Notebook for testing predictions
├── main.py # Python script version of the model
├── imdb_rnn_model.h5 # Trained model file (HDF5 format)
├── imdb_rnn_model.keras # Trained model file (Keras format)
├── requirements.txt # List of dependencies
└── .gitignore # Files ignored by Git


---

## 🧠 Model Architecture

- **Embedding Layer**: Converts word indices into dense vectors of fixed size.
- **SimpleRNN Layer**: Processes the sequence data.
- **Dense Layer**: Outputs binary sentiment classification using sigmoid activation.

```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

Source: IMDB Reviews Dataset

Size: 25,000 training samples and 25,000 test samples

Classes: Positive (1) or Negative (0)

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harsha-chichu/Movie_Classification_RNN.git
cd Movie_Classification_RNN

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

###3. Install dependencies

```bash
pip install -r requirements.txt

###4. Run the model

```bash
You can use either notebooks or the Python script:

Open simplernn.ipynb to train and evaluate the model.

Open prediction.ipynb to test predictions.

Or run main.py for a script-based workflow.





