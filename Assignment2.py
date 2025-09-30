import nltk
from nltk.tokenize import word_tokenize


def extract_features(sentence):
    """
    Extract features for the occurrence of 'it' in a sentence.

    Args:
        sentence (str): The sentence containing 'it'

    Returns:
        dict: Dictionary of features
    """
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())

    # F1: Position of "it" in the sentence (1-indexed)
    it_position = None
    for i, token in enumerate(tokens, start=1):
        if token == 'it':
            it_position = i
            break

    return {
        'F1_position': it_position
    }


def process_corpus(file_path):
    """
    Process the it-corpus.tsv file and extract features for each sentence.

    Args:
        file_path (str): Path to the TSV file

    Returns:
        list: List of dictionaries containing features for each sentence
    """
    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                anaphoric_class = parts[0]
                sentence = parts[1]

                features = extract_features(sentence)
                results.append({
                    'class': anaphoric_class,
                    'sentence': sentence,
                    'features': features
                })

    return results


if __name__ == '__main__':
    # Process the corpus
    results = process_corpus('it-corpus.tsv')

    print("F1 results:")
    for i, result in enumerate(results, start=1):
        print(f" {i} :  F1 (Position of 'it'): {result['features']['F1_position']}")
