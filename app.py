# app.py
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_loader import get_word_index

app = Flask(__name__)
model = load_model('sentiment_model.h5')
word_index = get_word_index()

def preprocess_review(review, maxlen=200, num_words=10000):
    tokens = review.lower().split()
    # Debug: Print tokens
    # print(f"Tokens: {tokens}")
    # Map words to indices, filter out-of-vocab words
    sequence = [word_index.get(word, 0) for word in tokens if word_index.get(word, 0) < num_words]
    # print(f"Sequence before padding: {sequence}")
    # Pad sequence
    padded = pad_sequences([sequence], maxlen=maxlen)
    # print(f"Padded sequence: {padded}")
    # In preprocess_review()
    missing_words = [word for word in tokens if word_index.get(word, 0) >= num_words or word_index.get(word, 0) == 0]
    # print(f"Missing words (out of vocab): {missing_words}")
    return padded
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sequence = preprocess_review(review)
        prediction = model.predict(sequence, verbose=0)[0][0]
        # print(f"Raw prediction score: {prediction}")
        
        # Define thresholds for Positive, Neutral, Negative
        if prediction > 0.6:
            result = 'Positive'
        elif prediction >= 0.5:
            result = 'Neutral'
        else:
            result = 'Negative'
        
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)