# main.py
from data_loader import load_and_preprocess_data
from model import build_model, train_model
from visualize import plot_history  # Import the plotting function

def main():
    # Load data
    (train_data, train_labels), (test_data, test_labels) = load_and_preprocess_data(maxlen=200, num_words=10000)
    
    # Build and train model
    model = build_model(maxlen=200)
    history = train_model(model, train_data, train_labels, epochs=10, batch_size=128)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('sentiment_model.h5')
    print("Model saved as 'sentiment_model.h5'")
    
    # Plot results
    plot_history(history)

if __name__ == "__main__":
    main()