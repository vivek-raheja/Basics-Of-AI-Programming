# Vivek Raheja
# ITP 259 Spring 2024
# HW4


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



def main():


    N = 400
    theta = np.linspace(0, 2 * np.pi, N)
    r_a = 2 * theta + np.pi
    r_b = -2 * theta - np.pi

    # spiral A
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    labels_a = np.zeros((N, 1))

    # spiral B
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    labels_b = np.ones((N, 1))


    noise_level = 0.5

    # add noise
    data_a_noisy = data_a + np.random.normal(scale=noise_level, size=data_a.shape)
    data_b_noisy = data_b + np.random.normal(scale=noise_level, size=data_b.shape)

    data_noisy = np.vstack((data_a_noisy, data_b_noisy))
    labels = np.vstack((labels_a, labels_b)).flatten()
    shuffle_indices = np.random.permutation(data_noisy.shape[0])
    data_noisy, labels = data_noisy[shuffle_indices], labels[shuffle_indices]

    # splitting dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_noisy, labels, test_size=0.30, random_state=2023, stratify=labels
    )

    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = break4mint.transform(X_test)


    mlp = MLPClassifier(hidden_layer_sizes=(5, 5), activation="relu",
                        max_iter=1000, alpha=1e-3, solver="adam",
                        random_state=2023, learning_rate_init=0.01, verbose=True)
    mlp.fit(X_train, Y_train)

    # spiral distribution with noise
    plt.figure(figsize=(8, 6))
    plt.scatter(data_noisy[:, 0], data_noisy[:, 1], c=labels, cmap='viridis')
    plt.title('Spiral Plot with Noise')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(mlp.loss_curve_)
    plt.title('Loss Curve for MLP Classifier')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    Y_pred = mlp.predict(X_test)

    # confusion matrix
    conf_mat = confusion_matrix(Y_test, Y_pred)

    plot_labels = ['No', 'Yes']
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=plot_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # accuracy
    print("The accuracy is", mlp.score(X_test, Y_test))


    # spirals with classes
    x_min, x_max = data_noisy[:, 0].min() - 1, data_noisy[:, 0].max() + 1
    y_min, y_max = data_noisy[:, 1].min() - 1, data_noisy[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

    plt.scatter(data_noisy[:, 0], data_noisy[:, 1], c=labels, edgecolors='k', cmap='viridis')
    plt.title('Decision Boundary with Spirals')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

main()