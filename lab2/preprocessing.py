import pandas as pd
import tensorflow as tf
import numpy as np

# Load and preprocess data

# %%

# Load captions from file and convert it to dataframe

def load_data(data_path="lab2/dataF8K/captions.txt"):
    return pd.read_csv(data_path, header=0)


captions = load_data()
print("number of captions:", captions.shape[0])
captions.head()

# %%

# Add unique start word ">>>" (stop word = "." already present in captions)

captions["caption_with_start"] = ">>> " + captions["caption"].str.lower()
captions.head()

# %%

# Tokenize captions, keeping all words that appear at least 5 times

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # don't filter punctuation
tokenizer.fit_on_texts(captions["caption_with_start"])

count_thres = 5
low_count_words = [w for w, c in tokenizer.word_counts.items() if c < count_thres]
for w in low_count_words:  # remove words with low threshold
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]

captions["token_caption"] = tokenizer.texts_to_sequences(captions["caption_with_start"])

captions.head()

#%%
# One hot encoding captions

vocab_size = len(tokenizer.word_counts) # get number of words in the vocabulary list
caption = np.array(captions["caption"]) # turn captions into numpy arrays
caption_tok = tokenizer.texts_to_sequences(caption) # tokenize the captions

captions_oh =[] # one hot encoded tokenized captions
for c in caption_tok:
    oh_tensor = tf.one_hot(np.array(c), vocab_size)
    captions_oh.append(np.array(oh_tensor))

#%%
# Maximum length of a caption

max_length = len(captions_oh[0])
for caption_oh in captions_oh:
    length = len(caption_oh)
    if max_length < length:
        max_length= length
print(max_length)

print(vocab_size)
