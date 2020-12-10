import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from model import cnn_model
from matplotlib import pyplot as plt


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

# %%
# One hot encoding captions

vocab_size = len(tokenizer.word_counts)  # get number of words in the vocabulary list
caption = np.array(captions["caption"])  # turn captions into numpy arrays
caption_tok = tokenizer.texts_to_sequences(caption)  # tokenize the captions

captions_oh = []  # one hot encoded tokenized captions
for c in caption_tok:
    oh_tensor = tf.one_hot(np.array(c), vocab_size)
    captions_oh.append(np.array(oh_tensor))

# %%
# Maximum length of a caption

max_length = len(captions_oh[0])
for caption_oh in captions_oh:
    length = len(caption_oh)
    if max_length < length:
        max_length = length

print('This is the max length of a caption', max_length)
print('This is the total number of words used in the captions', vocab_size)

# %%

# Prepare images through CNN model : returns a dict img_features, the predicted values by the cnn model

img_features = dict()
# Extract features from each photo
image_path = "lab2/dataF8K/Images"

for name in tqdm(os.listdir(image_path)):
    filename = image_path + '/' + name
    image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    # Convert the image pixels to a numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)
    # Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Prepare the image for the CNN Model model
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    # Pass image into model to get encoded features
    feature = cnn_model.predict(image, verbose=0)
    # Store encoded features for the image
    image_id = name.split('.')[0]
    img_features[image_id] = feature

# {'2387197355_237f6f41ee': array([[5.52348602e-08, 1.53278805e-07, 8.60161094e-07, 1.02808038e-07,
#         1.64723151e-08, 4.35574918e-08, 4.02927007e-08, 2.50875463e-08,
#         7.48211875e-08, 3.58235397e-08, 9.00228443e-08, 1.31862691e-07,
#         1.15471570e-08, 6.80091432e-08, 3.30712524e-08, 8.98553267e-08,....}



# %%
## Test
##img = tf.keras.preprocessing.image.load_img("lab2/dataF8k/Images/667626_18933d713e.jpg", target_size=(299, 299))
##img = tf.keras.preprocessing.image.img_to_array(img)
##img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
##img = tf.keras.applications.inception_v3.preprocess_input(img)
##prediction = cnn_model.predict(img, verbose=0)
##print(prediction)

# %%
