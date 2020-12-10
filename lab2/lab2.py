import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import random
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

# %%

# Encoder Model = Pretrained InceptionV3 CNN model

cnn_model = InceptionV3()

cnn_model._layers.pop()  # remove last layer for our encoder (classification layer)
cnn4nw = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-1].output)

print("Output shape:", cnn4nw.layers[-1].output_shape)


# %%

# Decoder Model = RNN model

def RNNModel():
    # Some parameters of the model
    vocab_size = 3002  # total number of words in the vocabulary
    max_len = 40  # maximum number of words used in a caption (it is actually 37)
    embedding_size = 512  # size of the embedding layer used as input for the LSTMs
    output_cnn_size = 2048  # size of the output vector of the CNN model InceptionV3
    size = 600

    # Pipeline 1 : Layers after the CNN output
    image_input = tf.keras.layers.Input(
        shape=(output_cnn_size,))  # an input layer which size is the output size of the cnn model
    image_model_1 = tf.keras.layers.Dropout(0.3)(image_input)  # a dropout layer for optimization
    image_model = tf.keras.layers.Dense(embedding_size, activation='relu')(
        image_model_1)  # A dense layer used for embedding the output vectors of the CNN model

    # Pipeline 2 : Layers after mapping words to one hot vectors
    caption_input = tf.keras.layers.Input(shape=(max_len,))
    caption_model_1 = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
    caption_model_2 = tf.keras.layers.Dropout(0.3)(caption_model_1)
    caption_model = tf.keras.layers.LSTM(size)(caption_model_2)

    # Merging the models and creating a softmax classifier
    final_model_1 = tf.keras.layers.concatenate([image_model, caption_model])
    final_model_2 = tf.keras.layers.Dense(size, activation='relu')(final_model_1)
    final_model = tf.keras.layers.Dense(vocab_size, activation='softmax')(final_model_2)

    # Creating the model
    rnn_model = tf.keras.models.Model(inputs=[image_input, caption_input], outputs=final_model)
    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return rnn_model


# %%
# Load and preprocess data

# Load captions from file and convert it to dataframe

def load_data(data_path="lab2/dataF8K/captions.txt"):
    return pd.read_csv(data_path, header=0)


data = load_data()
print("number of captions:", data.shape[0])
# data.head()

# %%

# Add unique start word ">>>" (stop word = "." already present in captions)

data["caption_with_start"] = ">>> " + data["caption"].str.lower()
# data.head()

# %%

# Tokenize captions, keeping all words that appear at least 5 times

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # don't filter punctuation
tokenizer.fit_on_texts(data["caption_with_start"])

count_thres = 5
low_count_words = [w for w, c in tokenizer.word_counts.items() if c < count_thres]
for w in low_count_words:  # remove words with low threshold
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]

data["token_caption"] = tokenizer.texts_to_sequences(data["caption_with_start"])

# data.head()
# print(data["token_caption"][0])

# %%
captions = np.array(data["token_caption"])
# print(captions)
# print(captions[0])

# %%

# Prepare images through CNN model
images = data["image"].to_numpy()
def encode_images(images, length=None):
    images_enc = []
    if length is None:
        length = len(images) - 1
    for i in tqdm(range(length)):
        if i == 0 or images[1] != images[i - 1]:
            image_path = "lab2/dataF8k/Images/" + images[i]
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
            img = tf.keras.preprocessing.image.img_to_array(img)
            x = img.reshape((1,) + img.shape)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            prediction = cnn4nw.predict(x)
            prediction = prediction.reshape(2048)
            images_enc.append(prediction)
        else:
            images_enc.append(images_enc[i - 1])
    return images_enc


#images_enc = encode_images(images)
"""
	images_enc is of form:
	[array([[0.12277617, 0.33294916, 0.75271696, ..., 0.30216402, 0.4028324 ]], dtype=float32) # Image 1 
	array([[0.12277617, 0.33294916, 0.75271696, ..., 0.30216402, 0.4028324 ]], dtype=float32) # Image 1

"""
#%%

images_enc = np.load('lab2/images_encoded.npy')
print(len(images_enc))
print(images_enc[0])


# %%
# captions_for_test = captions[:500]

# print(images_enc_test)  # 500 captions
# [array([[0.12277617, 0.33294916, 0.75271696, ..., 0.21939668, 0.30216402,
#         0.4028324 ]], dtype=float32), array([[0.12277617, 0.33294916, 0.75271696, ..., 0.21939668, 0.30216402,
#         0.4028324 ]], dtype=float32), array([[0.12277617, 0.33294916, 0.75271696, ..., 0.21939668, 0.30216402,...]

# print(images_enc_test[0].shape)  # (1, 2048)
# print(len(images_enc_test))  # 500
# %%
# print(images_enc_test[0])
# print(captions_test)  # 500 captions

# %%
# Test captions and images
# first_caption = captions_test[0]
# first_image = images_enc_test[0]
# print(first_caption)
# print(first_image)


# %%
# Create sequences of images, input sequences and output words for a single caption

def create_sequences(caption, image, max_length=40, vocab_size=3002):
    # X1 : input for image features
    # X2 : input for text features
    # y  : output word
    X1, X2, y = list(), list(), list()
    for i in range(1, len(caption)):
        in_seq, out_seq = caption[:i], caption[i]
        in_seq_padded = np.zeros(max_length)
        in_seq_padded[:len(in_seq)] = in_seq
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X1.append(image)
        X2.append(in_seq_padded)
        y.append(out_seq)
    return X1, X2, y


# print(create_sequences(first_caption, first_image)[0])


# %%

# Gives all captions and images as input and creates X1, X2, y for batches
# X1 shape : sum(caption in batch) [caption length - 1] rows x 2048 columns
# X2 shape:  sum(caption in batch) [caption length - 1] rows x max_length columns
# y shape: sum(caption in batch) [caption length - 1] rows x vocab_size columns
def data_generator(captions, images, batch_size, max_length=40, vocab_size=3002, random_seed=10):
    random.seed(random_seed)
    print('here')
    # Setting random seed for reproducibility of results
    index_list = np.arange(len(captions))
    _count = 0
    assert batch_size <= len(captions), 'Batch size must be less than or equal to {}'.format(len(images))
    while True:
        if _count >= len(captions):
            # Generator exceeded or reached the end so restart it
            _count = 0
            random.shuffle(index_list)
        # Batch list to store data
        input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
        for i in range(_count, min(len(captions), _count + batch_size)):
            # Retrieve the image id
            index = index_list[i]
            # Retrieve the image features
            image = images[index]
            # Retrieve the captions list
            caption = captions[index]
            input_img, input_sequence, output_word = create_sequences(caption, image, max_length, vocab_size)
            # Add to batch
            for j in range(len(input_img)):
                input_img_batch.append(input_img[j])
                input_sequence_batch.append(input_sequence[j])
                output_word_batch.append(output_word[j])
        _count = _count + batch_size
        yield ([np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch))


# %%
batch_size = 32
size_train = int(0.8 * len(captions))
captions_train, captions_val = captions[:size_train], captions[size_train:]
images_train, images_val = images[:size_train], images[size_train:]
# %%
generator_train = data_generator(captions_train, images_train, batch_size)
generator_val = data_generator(captions_val, images_val, batch_size)

# %%
steps_per_epoch = len(captions) / batch_size
my_model = RNNModel()
my_model.fit_generator(generator_train, epochs=20, steps_per_epoch=steps_per_epoch, validation_data=generator_val)
