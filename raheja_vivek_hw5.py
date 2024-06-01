#Vivek Raheja
#ITP 259 Spring 2024
#HW5

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import seaborn as sb
from sklearn.model_selection import train_test_split

def main():
    numberData = pd.read_csv("A_Z Handwritten Data.csv")

    X = numberData.iloc[1:, 1:]
    y = numberData.iloc[1:, 0]

    print(X.shape)
    print(y.shape)

    # the target variables are numbers (0....25) that each represent a letter (A-Z)

    word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
     7: ' H', 8: 'I ', 9: 'J ', 10: ' K', 11: ' L', 12: ' M',
     13: ' N ', 14: ' 0', 15: ' P', 16: ' Q ', 17: ' R', 18: 'S',
     19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    numberData['label_letter'] = numberData['label'].map(word_dict)

    plt.figure(1)
    sb.countplot(x="label_letter", data=numberData, palette="Spectral", hue="label_letter")
    plt.show()

    mapping_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
     7: ' H', 8: 'I ', 9: 'J ', 10: ' K', 11: ' L', 12: ' M',
     13: ' N ', 14: ' 0', 15: ' P', 16: ' Q ', 17: ' R', 18: 'S',
     19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    plt.figure(figsize=(8, 8))
    for i in range(64):
        index = random.randint(0, len(X))
        plt.subplot(8, 8, i+1)
        plt.imshow(X.iloc[index].values.reshape(28, 28), cmap='gray')
        plt.title(mapping_dict[y.iloc[index]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    mlp = MLPClassifier(random_state=2023, hidden_layer_sizes=(100, 100, 100), max_iter=25)
    mlp.fit(X_train_scaled, y_train)
    plt.plot(mlp.loss_curve_)
    plt.show()

    print("The accuracy is", mlp.score(X_test, y_test))

    y_pred = mlp.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()

    first_sample = X_test.iloc[0].values.reshape(1, -1)
    predicted_label = mlp.predict(first_sample)[0]
    image = first_sample.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    actual_label = y_test.iloc[0]
    plt.title(f"Predicted: '{chr(predicted_label + 65)}', Actual: '{chr(actual_label + 65)}'")
    plt.axis('off')
    plt.show()

    failed_df = X_test[y_pred != y_test].reset_index(drop=True)
    failed_y_test = y_test[y_pred != y_test].reset_index(drop=True)
    failed_idx = failed_df.sample(n=1).index.item()
    failed_sample = failed_df.iloc[failed_idx].values.reshape(28, 28)
    plt.imshow(failed_sample, cmap='gray')
    actual_label = failed_y_test.iloc[failed_idx]
    predicted_label = y_pred[failed_idx]
    plt.title(f"The failed predicted letter is '{chr(predicted_label + 65)}' whereas the actual letter is '{chr(actual_label + 65)}'")
    plt.axis('off')
    plt.show()

main()