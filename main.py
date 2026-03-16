# main.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create KNN model
    k = 3
    model = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"KNN Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
    