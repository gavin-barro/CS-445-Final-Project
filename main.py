from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import csv
import numpy as np
# import tensorflow as tf


def read_csv_into_numpy(filename: str, max_lines: int = 1000) -> tuple:
    data = []
    seventh_column = []
    # Getting rid of the string columns that aren't relevant to our data
    excluded_columns = [1, 2, 5, 6, 7, 11]
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            if line_count >= max_lines:
                break
            filtered_row = [float(row[i]) for i in range(len(row)) if i not in excluded_columns]
            data.append(filtered_row)  # Excluding the 7th column
            seventh_column.append(float(row[6]))   # Storing only the 7th column
            line_count += 1

    return np.array(data), np.array(seventh_column)


def main() -> None:
    # TODO use the csv files to run predictions on this
    path = "Eartquakes-1990-2023.csv"
    X, y = read_csv_into_numpy(path)

    # print(f"X: {X}")
    # print(f"y: {y}")

    # Perform standard scaling for X
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Perform standard scaling for y
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


    # Singular decision tree
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state = 100)
    tree = DecisionTreeRegressor()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE - Decision Tree: {mse}")



    # Random Forest
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE - Random Forest: {mse}")



    # SVM
    svm_regressor = SVR(kernel='linear')
    svm_regressor.fit(X_train, y_train)
    y_pred = svm_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE - SVM:", mse)


    # # Neural Network
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 100)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation = "relu"),
    #     tf.keras.layers.Dense(32, activation = 'relu'),
    #     tf.keras.layers.Dense(1)
    # ])

    # model.compile(optimizer='adam', loss='mean_squared_error')

    # history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # # Make predictions on the testing set
    # y_pred = model.predict(X_test)

    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # print("MSE - NN: ", mse)


    # Plot it using matplotlib





if __name__ == "__main__":
    main()
