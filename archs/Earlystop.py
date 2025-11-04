class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
       self.metrics_results = []
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.model_weights = None
    def get_best_weights(self):
        return self.model_weights
    def __call__(self, val_loss, model_weights):
        if len(self.metrics_results) == 0:
            self.metrics_results.append(val_loss)
            self.model_weights = model_weights
            return False
        else:
            if val_loss < min(self.metrics_results) - self.min_delta:
                self.metrics_results.append(val_loss)
                self.counter = 0
                self.model_weights = model_weights
                return False
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered")
                    return True
                return False