# train_and_save_model.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import pickle

# Load dataset and train a KNN model
iris = load_iris()
X, y = iris.data, iris.target
model = KNeighborsClassifier()
model.fit(X, y)

# Save the model to a pickle file
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)
