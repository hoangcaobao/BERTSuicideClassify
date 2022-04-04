import numpy as np
from clean_data import *
from prepare import *
from save_history import *
from model import *
from confusion import *
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", default="Suicide_Detection.csv", type=str)
    parser.add_argument("--text-title", default="text", type=str)
    parser.add_argument("--label-title", default="class", type=str)
    parser.add_argument("--label-category", default="suicide", type=str)
    parser.add_argument("--weights-path", default="bert.h5", type=str)
    parser.add_argument("--history-image", default="history.png", type=str)
    parser.add_argument("--history-csv", default="history.csv", type=str)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    
    args=parser.parse_args()
    
    texts, labels = csv_to_data(args.data_path, args.text_title, args.label_title, args.label_category)
    texts, temp1, labels, temp2 = split(texts, labels, 0.7)
    
    clean(texts)
    x_train, x_valid, y_train, y_valid=split(texts, labels, args.test_size)
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    x_test = temp1[:5000]
    y_test = np.array(temp2[:5000])
    clean(x_test)
    x_test = np.array(x_test)

    model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.weights_path,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    verbose=1)
    
    model = build_model("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2")

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(args.lr), metrics=['accuracy'])
    history=model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_valid,y_valid), callbacks=[model_callback])
    preds = model.predict(x_test).argmax(axis=-1)
    save_plot_and_csv(history, args.history_csv, args.history_image)
    save_confusion_matrix(y_test, preds)
