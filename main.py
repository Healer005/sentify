# main.py
from data_loader import load_and_preprocess_data
from model import build_model, train_model
from sklearn.metrics import classification_report
import numpy as np

def main():
    # Load data
    (train_data, train_labels), (test_data, test_labels) = load_and_preprocess_data(maxlen=200, num_words=10000)
    
    # Build and train model
    model = build_model(maxlen=200)
    history = train_model(model, train_data, train_labels, epochs=15, batch_size=128)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Predict on test set
    predictions = model.predict(test_data)
    predicted_labels = (predictions > 0.5).astype(int).flatten()  # Threshold at 0.5
    print("Classification Report:")
    print(classification_report(test_labels, predicted_labels, target_names=['Negative', 'Positive']))
    
    # Save model
    model.save('sentiment_model.h5')

if __name__ == "__main__":
    main()