import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
import csv
from nltk import pos_tag, ne_chunk
def extract_features(sentence):
    """
    Extract features for all occurrences of 'it' in a sentence.

    Args:
        sentence (str): The sentence containing 'it'

    Returns:
        list: List of positions (1-indexed) for each occurrence of 'it'
    """
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())

    # F1: Positions of all "it" occurrences in the sentence (1-indexed)
    it_positions = []
    for i, token in enumerate(tokens, start=1):
        if token == 'it':
            it_positions.append(i)

    return it_positions

def number_of_tokens(sentence):
    """
    Count the number of tokens in a sentence.

    Args:
        sentence (str): The sentence to be tokenized

    Returns:
        int: Number of tokens in the sentence
    """
    tokens = word_tokenize(sentence)
    return len(tokens)

def number_of_punctuation(sentence):
    """
    Count the number of punctuation marks in a sentence using NLTK POS tagging.

    Args:
        sentence (str): The sentence to be analyzed

    Returns:
        int: Number of punctuation marks in the sentence
    """
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    # Count tokens tagged as punctuation (POS tags starting with punctuation symbols)
    punctuation_count = sum(1 for word, pos in pos_tags if pos in ['.', ',', ':', ';', '!', '?', '-', '--', '...', "''", '``', '(', ')', '[', ']', '{', '}'])

    return punctuation_count


def count_preceding_noun_phrases(sentence, it_position):
    """
    Count the number of atomic noun phrases that come before a specific instance of 'it'.

    Args:
        sentence (str): The sentence to be analyzed
        it_position (int): The position of 'it' in the sentence (1-indexed)

    Returns:
        int: Number of noun phrases preceding 'it'
    """
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # Use chunking to identify noun phrases
    # Define a simple grammar for atomic noun phrases
    grammar = r"""
        NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
    """
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    # Count noun phrases that appear before the 'it' position
    np_count = 0
    token_index = 0

    for subtree in tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
            # Get the position of the last token in this NP
            np_end_position = token_index + len(subtree)
            # If this NP ends before our 'it' position, count it
            if np_end_position < it_position:
                np_count += 1
            token_index += len(subtree)
        else:
            token_index += 1

    return np_count

def count_following_noun_phrases(sentence, it_position):
    '''
    Count the number of atomic noun phrases that come after a specific instance of "it"

    Args:
        sentence (str): The sentence to ne analyzed
        it_position (int): The positon of "it" in the sentence (1-indexed)
    
    Returns:
        int: Number of noun phrases following "it"
    '''
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    grammar = r"""
        NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
    """
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    np_count = 0
    token_index = 0

    for subtree in tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
            np_end_position = token_index + len(subtree)
            if np_end_position > it_position:
                np_count += 1
            token_index += len(subtree)
        else:
            token_index += 1
    
    return np_count

def follows_prepositional_phrase(sentence, it_position):
    '''
    Determines whether a specific instance of "it" immediately follows a prepositional phrase

    Args:
        sentence(str): The sentence to be analyzed
        it_positon(int): The positon of "it" in the sentence (1-indexed)
    
    Returns:
        bool: True if "it" immediately follows a prepositional phrase, false otherwise
    '''
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    grammar = r"""
        PP:{<IN><DT|PRP\$>?<JJ>*<NN.*>+}
    """

    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    token_index = 0

    for subtree in tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'PP':
            pp_end_position = token_index + len(subtree)
            if pp_end_position == it_position:
                return True
            token_index += len(subtree)
        else:
            token_index += 1
    
    return False

def preceding_succeeding_pos_tags(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    before = [
        pos_tags[i][1] if i >= 0 else "ABS" for i in range(it_position-5, it_position-1)
    ]
    after = [
        pos_tags[i][1] if i < len(pos_tags) else "ABS" for i in range(it_position, it_position + 4)
    ]

    return before+after

def followed_by_ing_verb(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position < len(pos_tags): 
        if pos_tags[it_position][1] == 'VBG':
            return True
    return False

def followed_by_preposition(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position  < len(pos_tags):
        if pos_tags[it_position][1] == 'IN':
            return True
    return False

def num_adjectives_after(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    count_adj = 0
    for i in range(it_position, len(pos_tags)):
        if pos_tags[i][1] == 'JJ' or pos_tags[i][1] == 'JJR' or pos_tags[i][1] == 'JJS':
            count_adj += 1
    
    return count_adj 

def preceded_by_verb(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position-2 > 0:
        return pos_tags[it_position-2][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    return False

def followed_by_verb(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position < len(pos_tags):
        return pos_tags[it_position][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    return False

def followed_by_adj(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position < len(pos_tags):
        return pos_tags[it_position][1] in ['JJ', 'JJR', 'JJS']

    return False

def process_corpus(file_path):
    """
    Process the it-corpus.tsv file and extract features for each instance of 'it'.

    Args:
        file_path (str): Path to the TSV file

    Returns:
        list: List of dictionaries containing features for each instance of 'it'
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

                it_positions = extract_features(sentence)
                numTokens = number_of_tokens(sentence)
                numPunc = number_of_punctuation(sentence)

                # Create a separate row for each occurrence of 'it'
                for position in it_positions:
                    num_preceding_nps = count_preceding_noun_phrases(sentence, position)
                    num_following_nps = count_following_noun_phrases(sentence, position)
                    prepositional_phrase = follows_prepositional_phrase(sentence, position)
                    pos_tags_preceding_and_succeeding = preceding_succeeding_pos_tags(sentence, position)
                    ing_verb = followed_by_ing_verb(sentence, position)
                    preposition = followed_by_preposition(sentence, position)
                    adjectives_after = num_adjectives_after(sentence, position)
                    verb_preceding = preceded_by_verb(sentence, position)
                    verb_following = followed_by_verb(sentence, position)
                    adj_following = followed_by_adj(sentence, position)
                    results.append({
                        'class': anaphoric_class,
                        'f1_position_it': position,
                        'f2_num_tokens': numTokens,
                        'f3_num_punctuation': numPunc,
                        'f4_num_preceding_nps': num_preceding_nps,
                        'f5_num_following_nps': num_following_nps,
                        'f6_prepositional_phrase': prepositional_phrase,
                        'f7_pos_tags_preceding_and_succeeding': pos_tags_preceding_and_succeeding,
                        'f8_ing_verb': ing_verb,
                        'f9_preposition': preposition,
                        'f10_num_adjectives_after': adjectives_after,
                        'f11_verb_preceding': verb_preceding,
                        'f12_verb_following': verb_following,
                        'f13_adj_following': adj_following
                    })

    return results


if __name__ == '__main__':
    # Process the corpus
    results = process_corpus('it-corpus.tsv')

    
    

    # Export results to CSV
    with open('features_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['class', 'f1_position_it', 'f2_num_tokens', 'f3_num_punctuation', 'f4_num_preceding_nps', 'f5_num_following_nps', 'f6_prepositional_phrase', 'f7_pos_tags_preceding_and_succeeding', 'f8_ing_verb', 'f9_preposition',
                      'f10_num_adjectives_after', 'f11_verb_preceding', 'f12_verb_following', 'f13_adj_following']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"Features extracted and saved to features_output.csv ({len(results)} rows)")

