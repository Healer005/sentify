from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(maxlen=200, num_words=10000):
    """
    Load IMDB dataset and pad sequences to a fixed length.
    Args:
        maxlen (int): Max review length (words).
        num_words (int): Vocabulary size (top frequent words).
    Returns:
        Tuple: (train_data, train_labels), (test_data, test_labels)
    """
    # Load dataset: 25k training, 25k testing reviews
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    # Pad sequences so all reviews are 200 words long
    train_data = pad_sequences(train_data, maxlen=maxlen)
    test_data = pad_sequences(test_data, maxlen=maxlen)
    return (train_data, train_labels), (test_data, test_labels)

def get_word_index():
    """Get word-to-index mapping for decoding or preprocessing."""
    return imdb.get_word_index()