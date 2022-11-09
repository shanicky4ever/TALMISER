from sklearn.ensemble import RandomForestClassifier
import joblib
import logging


class modelHandler:
    def __init__(self, svm_config, pretrained_model_path=None):
        self.svm_config = svm_config
        if not pretrained_model_path:
            self.model = RandomForestClassifier(
                max_depth=svm_config['max_depth'])
        else:
            self._load_model(pretrained_model_path)

    def train(self, X_train, X_val, y_train, y_val):
        self.model.fit(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        logging.info(f'val acc is {val_acc}')
        self._save_model()

    def _save_model(self):
        joblib.dump(self.model, self.svm_config["weight_file"])

    def _load_model(self, pretrained_model_path):
        self.model = joblib.load(pretrained_model_path)

    def simple_forward(self, data):
        return self.model.predict(data).tolist()
