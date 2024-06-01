#Vivek Raheja
#ITP 259 Spring 2024
#HW3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler



def main():
    diabetes_knn = pd.read_csv("diabetes.csv")
    pd.set_option("display.max_columns", None)

    X = diabetes_knn.drop("Outcome", axis=1)
    y = diabetes_knn["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2023,
                                                                  stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=2023,
                                                      stratify=y_train_full)


    neighbors_settings = range(1, 31)
    trainingA_accuracy = []
    trainingB_accuracy = []

    for n_neighbors in neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        trainingA_accuracy.append(knn.score(X_train, y_train))
        trainingB_accuracy.append(knn.score(X_val, y_val))

    print(knn.score(X_train, y_train), knn.score(X_val, y_val))


    plt.title("KNN: Varying number of neighbors")
    plt.plot(neighbors_settings, trainingA_accuracy, label="TrainingA Accuracy")
    plt.plot(neighbors_settings, trainingB_accuracy, label="trainingB Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Neighbors")
    plt.legend()
    plt.show()

    best_k = 20  # total sample / 2 -> sqroot
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train_full, y_train_full)
    test_accuracy = knn_best.score(X_test, y_test)
    print(test_accuracy)

    cm = confusion_matrix(y_test, knn_best.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_best.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    new_person = np.array([[2, 150, 85, 22, 200, 30, 0.3, 55]])
    new_person_scaled = scaler.transform(new_person)
    prediction = knn_best.predict(new_person_scaled)
    print(prediction)


main()