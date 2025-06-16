from flask import Flask, request, jsonify
import fasttext
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

model = fasttext.load_model("stars_number_multiclass.bin")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route("/")
def home():
    return "<h1>Star Rating Prediction API</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    text = preprocess_text(text)
    prediction = model.predict(text)[0][0].replace('__label__', '')
    return jsonify({'stars': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
