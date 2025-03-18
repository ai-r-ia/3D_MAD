import torch
class EarlyStopping:
    def __init__(self, logger, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.logger = logger

    def __call__(self, loss, model, checkpoint_path):
        if loss < self.best_loss:
            torch.save(model.state_dict(), checkpoint_path)
            self.logger.info(f"Model improved. Saved to {checkpoint_path}")
            self.logger.info(f"loss decreased ({self.best_loss:.6f} --> {loss:.6f}).")
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
