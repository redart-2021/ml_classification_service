from sklearn import neighbors
import pickle
from datetime import date


class ActiveClassificator:
    def __init__(self, file_name):
        self.model = pickle.load(open(file_name, 'rb'))
        self.model_name = file_name

    def get_model_name(self):
        return self.model_name

    def predict(self, data):
        return self.model.predict(data)

    def fit(self, data, n_clusters=6, random_state=0):
        self.model = neighbors.KNeighborsClassifier(n_clusters=n_clusters, random_state=random_state).fit(data)
        pickle.dump(self.model, open(f"{date.today()}_classificator.model", 'wb'))

