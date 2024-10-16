import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class Error:
    def __init__(self, scaled, yhat, testY):
        self.scaled = scaled
        self.yhat = np.array(yhat)
        self.y = np.array(testY)
        self.inv_yhat = scaled.inverse_transform(self.yhat.reshape(-1, 1))
        self.inv_y = scaled.inverse_transform(self.y.reshape(-1, 1))

    def get_mae(self):
        mae = mean_absolute_error(self.inv_y.reshape(-1), self.inv_yhat.reshape(-1))
        return mae

    def get_rmse(self):
        rmse = np.sqrt(mean_squared_error(self.inv_y.reshape(-1), self.inv_yhat.reshape(-1)))
        return rmse

    def get_mape(self):
        mape = mean_absolute_percentage_error(self.inv_y.reshape(-1), self.inv_yhat.reshape(-1)) * 100
        return mape