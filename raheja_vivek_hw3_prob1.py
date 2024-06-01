#Vivek Raheja
#ITP 259 Spring 2024
#HW3

import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    titanicData = pd.read_csv("Titanic.csv")
    pd.set_option("display.max_columns", None)

    titanicData.drop('Passenger', axis=1, inplace=True)
    titanic_dummies = pd.get_dummies(titanicData, columns=['Class', 'Sex', 'Age'])


    X = titanic_dummies.drop('Survived', axis=1)
    y = titanic_dummies['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    print(train_acc)
    y_test_probas = model.predict_proba(X_test)

    skplt.metrics.plot_lift_curve(y_test, y_test_probas)
    plt.show()

    conf_mat = confusion_matrix(y_test, y_test_pred)

    labels = ['No', 'Yes']
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    new_observation = pd.DataFrame({
        'Class_1st':[0],
        'Class_2nd': [0],
        'Class_3rd': [1],
        'Class_Crew': [0],
        'Sex_Female':[0],
        'Sex_Male': [1],
        'Age_Adult':[1],
        'Age_Child': [0]
    }, columns=X_train.columns)


    predicted_survivability = model.predict(new_observation)
    print(predicted_survivability)


main()