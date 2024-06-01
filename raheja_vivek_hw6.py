#Vivek Raheja
#ITP 259
#HW5

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

def main():

    fashion = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion.load_data()

    flattened_images = train_images.reshape(train_images.shape[0], -1)
    raw_data = pd.DataFrame(flattened_images)

    raw_data.to_csv('fashion.csv', index=False)
    fashion_df = pd.read_csv('fashion.csv')

    # print(fashion_df)
    # data is [60000 rows x 784 columns], each row is one 28x28 image

    X_train = train_images
    y_train = train_labels
    X_test = test_images
    y_test = test_labels

    print("Training features shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test features shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

    # target variable value is numbers, hence we map to clothing

    label_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=np.arange(-0.5, 10, 1), rwidth=0.8)
    plt.xticks(np.arange(0, 10), labels=[label_map[x] for x in range(10)])
    plt.xlabel('Apparel')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(25):
        idx = np.random.randint(0, X_train.shape[0])
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[idx], cmap=plt.cm.binary)
        plt.xlabel(label_map[y_train[idx]])
    plt.show()

    X_train = X_train/255
    X_test = X_test/255

    model = Sequential()
    model.keras.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.summary()
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    plt.figure(figsize=(10, 5))


    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['accuracy'], label='Test Accuracy')

    plt.title('Model Test Loss and Accuracy')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

    first_test_image = np.expand_dims(X_test[0], 0)
    predictions = model.predict(first_test_image)
    predicted_label = np.argmax(predictions)

    #  first test image
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[0], cmap=plt.cm.binary)
    plt.title(f"Predicted: {label_map[predicted_label]}\nActual: {label_map[y_test[0]]}")
    plt.axis('off')
    plt.show()

    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # misclassified images
    misclassified_indices = np.where(predicted_labels != y_test)[0]
    first_misclassified_index = misclassified_indices[0]
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[first_misclassified_index], cmap=plt.cm.binary)
    plt.title(
        f"Predicted: {label_map[predicted_labels[first_misclassified_index]]}\nActual: {label_map[y_test[first_misclassified_index]]}")
    plt.axis('off')
    plt.show()

main()