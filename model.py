from sklearn.ensemble import RandomForestClassifier
import numpy as np

class SeismicClassifier:
    def __init__(self):
        self.model = None
        
    def build_model(self, n_estimators=100):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        return model
    
    def train(self, X, y, **kwargs):
        if self.model is None:
            self.model = self.build_model()
        
        # Reshape the input data if needed
        X = X.reshape(X.shape[0], -1)
        
        # Train the model and keep track of training scores
        self.model.fit(X, y)
        
        # Return a dictionary similar to Keras history
        train_score = self.model.score(X, y)
        history = {
            'accuracy': [train_score],
            'val_accuracy': [train_score],  # Using same score for demo
            'loss': [1 - train_score],
            'val_loss': [1 - train_score]
        }
        
        return history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        # Reshape the input data if needed
        X = X.reshape(X.shape[0], -1)
        
        # Get probabilities for each class
        probabilities = self.model.predict_proba(X)
        return probabilities
