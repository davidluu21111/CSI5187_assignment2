import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
import csv
from nltk import pos_tag
import spacy

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")
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
        sentence (str): The sentence to be analyzed
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
            np_start_position = token_index + 1
            if np_start_position > it_position:
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
            #print(pp_end_position, it_position)
            if pp_end_position == it_position + 1:
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

def np_after_it_contains_adj(sentence, it_position):
    
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # Define grammar for noun phrases that can contain adjectives
    grammar = r"""
        NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
    """
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    token_index = 0

    for subtree in tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
            np_start_position = token_index + 1  # Convert to 1-indexed

            # Check if this NP comes after 'it'
            if np_start_position > it_position:
                # Check if this NP contains an adjective
                for _, pos in subtree:
                    if pos in ['JJ', 'JJR', 'JJS']:
                        return True  # Return True as soon as we find any NP with adjective

            token_index += len(subtree)
        else:
            token_index += 1

    return False  # No NP after 'it' contains an adjective

def tokens_before_infinitive(sentence, it_position):

    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    # Search for first infinitive (TO + VB) in the entire sentence
    for i in range(0, len(pos_tags) - 1):  # -1 because we need to check i+1
        if pos_tags[i][1] == 'TO' and pos_tags[i+1][1] == 'VB' and i >= it_position:
            return i

    return 0

def tokens_between_it_and_preposition(sentence, it_position):

    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    for i in range(it_position, len(pos_tags)):
        if pos_tags[i][1] == 'IN':
            return i - it_position

    return 0

def adj_np_after_it(sentence, it_position):
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    grammar = r"""
        ADJNP: {<JJ|JJR|JJS>+<DT|PRP\$>?<JJ>*<NN.*>+}
    """

    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    token_index = 0
    for subtree in tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'ADJNP':
            np_start_position = token_index + 1
            if np_start_position > it_position:
                return True
    
    return False


#  def adj_np_after_it(sentence, it_position):

#     tokens = word_tokenize(sentence)
#     pos_tags = nltk.pos_tag(tokens)

#     grammar = r"""
#         NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
#     """
#     cp = nltk.RegexpParser(grammar)
#     tree = cp.parse(pos_tags)

#     token_index = 0
#     found_adj = False

#     for subtree in tree:
#         if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
#             np_start_position = token_index + 1

#             if np_start_position > it_position:
#                 for _, pos in subtree:
#                     if pos in ['JJ', 'JJR', 'JJS']:
#                         return True

#             token_index += len(subtree)
#         else:
#             if token_index + 1 > it_position and pos_tags[token_index][1] in ['JJ', 'JJR', 'JJS']:
#                 found_adj = True
#             token_index += 1

#     return False 

def dependency_relation_type(sentence, it_position):
    
    # Parse the sentence with spaCy
    doc = nlp(sentence)

    # Convert NLTK tokens to find the correct 'it' token in spaCy
    nltk_tokens = word_tokenize(sentence.lower())

    it_count = 0
    target_it_index = None
    for i, token in enumerate(nltk_tokens):
        if token == 'it':
            it_count += 1
            if it_count == it_position:  # This assumes it_position counts which 'it' we want
                target_it_index = i
                break

    if target_it_index is None:
        target_it_index = it_position - 1

    
    it_token = None
    spacy_idx = 0
    for nltk_idx, nltk_tok in enumerate(nltk_tokens):
        # Find corresponding spaCy token
        if spacy_idx < len(doc):
            spacy_tok = doc[spacy_idx]
            # Match tokens (case-insensitive)
            if nltk_tok.lower() == spacy_tok.text.lower():
                if nltk_idx == target_it_index:
                    it_token = spacy_tok
                    break
                spacy_idx += 1
            else:
                # Handle tokenization differences
                spacy_idx += 1

    
    if it_token is None:
        it_tokens = [token for token in doc if token.text.lower() == 'it']
        if it_tokens:
            # Use position to select the right 'it' if multiple exist
            it_index = min(it_position - 1, len(it_tokens) - 1)
            it_token = it_tokens[it_index]

    if it_token is None:
        return "NONE"

    # Check if 'it' is a dependent 
    if it_token.dep_ == 'ROOT':
        return "NONE"

    
    return it_token.dep_

def is_weather_verb_following(sentence, it_position):

    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position < len(pos_tags):
        if pos_tags[it_position][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            verb = pos_tags[it_position][0].lower()
            try:
                from nltk.corpus import wordnet as wn
                synsets = wn.synsets(verb, pos=wn.VERB)
                for synset in synsets:
                    if synset.lexname() == 'verb.weather':
                        return True
            except:
                pass
    return False

def is_cognitive_verb_following(sentence, it_position):

    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    if it_position < len(pos_tags):
        if pos_tags[it_position][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            verb = pos_tags[it_position][0].lower()
            try:
                from nltk.corpus import wordnet as wn
                synsets = wn.synsets(verb, pos=wn.VERB)
                for synset in synsets:
                    if synset.lexname() == 'verb.cognition':
                        return True
            except:
                pass
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
                    np_contains_adj = np_after_it_contains_adj(sentence, position)
                    tokens_before_inf = tokens_before_infinitive(sentence, position)
                    tokens_to_prep = tokens_between_it_and_preposition(sentence, position)
                    adj_np_following = adj_np_after_it(sentence, position)
                    dep_relation = dependency_relation_type(sentence, position)
                    weather_verb = is_weather_verb_following(sentence, position)
                    cognitive_verb = is_cognitive_verb_following(sentence, position)
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
                        'f13_adj_following': adj_following,
                        'f14_np_after_it_contains_adj': np_contains_adj,
                        'f15_tokens_before_infinitive': tokens_before_inf,
                        'f16_tokens_between_it_and_prep': tokens_to_prep,
                        'f17_adj_np_after_it': adj_np_following,
                        'f18_dependency_relation': dep_relation,
                        'f19_weather_verb': weather_verb,
                        'f20_cognitive_verb': cognitive_verb
                    })

    return results


if __name__ == '__main__':
    # Process the corpus
    results = process_corpus('it-corpus.tsv')

    
    

    # Export results to CSV
    with open('features_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['class', 'f1_position_it', 'f2_num_tokens', 'f3_num_punctuation', 'f4_num_preceding_nps', 'f5_num_following_nps', 'f6_prepositional_phrase', 'f7_pos_tags_preceding_and_succeeding', 'f8_ing_verb', 'f9_preposition',
                      'f10_num_adjectives_after', 'f11_verb_preceding', 'f12_verb_following', 'f13_adj_following', 'f14_np_after_it_contains_adj', 'f15_tokens_before_infinitive', 'f16_tokens_between_it_and_prep', 'f17_adj_np_after_it', 'f18_dependency_relation', 'f19_weather_verb', 'f20_cognitive_verb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"Features extracted and saved to features_output.csv ({len(results)} rows)")

