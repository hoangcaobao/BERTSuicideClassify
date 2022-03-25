import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


def build_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    #model architecture
    data_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    x = hub.KerasLayer(tfhub_handle_preprocess)(data_input)
    x = hub.KerasLayer(tfhub_handle_encoder, trainable=True)(x)
    x = x['pooled_output']
    x = tf.keras.layers.Dropout(0.2)(x)
    data_output = tf.keras.layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(data_input, data_output)
    