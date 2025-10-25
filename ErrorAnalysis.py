import csv
import nltk

# Download NLTK resources if not already available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Input and output file paths
input_tsv = "it-corpus.tsv"
output_tsv = "pos_tags.tsv"

# If your TSV has a header and a specific column for sentences, set this:
sentence_column = "Sentence"  

# Read the TSV and write the tagged output
with open(input_tsv, "r", encoding="utf-8") as infile, open(output_tsv, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.DictReader(infile, delimiter="\t")
    fieldnames = reader.fieldnames + ["pos_tags"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for row in reader:
        text = row[sentence_column].strip() if sentence_column else list(row.values())[0].strip()
        if not text:
            row["pos_tags"] = ""
        else:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            tagged_sentence = " ".join([f"{word}/{tag}" for word, tag in pos_tags])
            row["pos_tags"] = tagged_sentence

        writer.writerow(row)

print(f"POS tags saved to {output_tsv}")



