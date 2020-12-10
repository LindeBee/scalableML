import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3

#%%

# Encoder Model = Pretrained VGG16 CNN model

cnn_model = InceptionV3()


#%%

# Decoder Model = RNN model

# Some parameters of the model
vocab_size = 3002  #total number of words in the vocabulary
max_len = 40 # maximum number of words used in a caption (it is actually 37)
embedding_size = 512 # size of the embedding layer used as input for the LSTMs
output_cnn_size = 1920 # size of the output vector of the CNN model ImageNet
size = 600

# Pipeline 1 : Layers after the CNN output
image_input = tf.keras.layers.Input(shape=(output_cnn_size,)) # an input layer which size is the output size of the cnn model
image_model_1 = tf.keras.layers.Dropout(0.3)(image_input) # a dropout layer for optimization
image_model = tf.keras.layers.Dense(embedding_size, activation='relu')(image_model_1) # A dense layer used for embedding the output vectors of the CNN model


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
#%%
