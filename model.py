from tensorflow.keras import models, layers

def build_model(maxlen, vocab_size=10000):
    """
    Build a sentiment analysis model with Embedding and LSTM.
    Args:
        maxlen (int): Input sequence length.
        vocab_size (int): Vocabulary size.
    Returns:
        Compiled Keras model.
    """
    model = models.Sequential([
        layers.Embedding(vocab_size, 32, input_length=maxlen),  # Converts words to vectors
        layers.LSTM(32),  # Processes sequences, remembers context
        layers.Dense(1, activation='sigmoid')  # Outputs 0 (negative) or 1 (positive)
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

def train_model(model, train_data, train_labels, epochs=10, batch_size=128):
    """
    Train the model.
    Returns:
        History object with training metrics.
    """
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history