import pickle
import time
import joblib
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.optimizer_v2.gradient_descent import SGD
from model.dynamic import Dynamic
from model.init_model import Attention


class Expert_PLUS_EnsembleModel:
    def __init__(self, slide_window, n_vars, k, class_patv):
        self.V_m = None
        self.L_m = None
        self.X_m = None
        self.alpha = tf.Variable(initial_value=0.25, dtype=tf.float32)
        self.beta = tf.Variable(initial_value=0.25, dtype=tf.float32)
        self.gmma = tf.Variable(initial_value=0.25, dtype=tf.float32)
        self.delta = tf.Variable(initial_value=0.25, dtype=tf.float32)
        self.optimizer = SGD(lr=0.001)
        self.vars = n_vars
        self.patience = 30
        self.X = None
        self.L = None
        self.V = None
        self.k = k
        self.threshold = 0.0001
        self.slide_window = slide_window
        self.E = None
        self.init_sub_model(class_)

    def init_sub_model(self, class_):
        self.X_m = joblib.load('model_fit/%s' % (class_) + '/xgb_model.pkl')
        self.L_m = load_model('model_fit/%s' % (class_) + '/lstm_model.h5', custom_objects={'Attention': Attention})
        self.V_m = joblib.load('model_fit/%s' % (class_) + '/bayesian_ridge_model.pkl')

    def expert_gate(self, x):
        self.X = self.X_m.predict(x)
        self.L = self.L.reshape(-1)
        return self.alpha * self.X + self.beta * self.L

    def mixer_gate(self, x):
        self.E = self.expert_gate(x)
        self.V = self.V_m.predict(x)
        return self.gmma * self.E + self.delta * self.V

    def adaptive_forecasting(self, x):
        E = self.expert_gate(x)
        M = self.mixer_gate(x)
        return E + M

    def sliding_window_loss_threshold(self, loss, windows_size=10, threshold=0.00005):
        if len(loss) < windows_size:
            return False
        window_average = sum(loss[-windows_size:]) / windows_size

        if loss[-1] + threshold >= window_average:
            self.patience -= 1
        else:
            self.patience = 30
        return self.patience <= 0

    def save_model(self, class_patv):
        self.alpha.numpy().tofile(f"model_fit/%s" % (class_) + "/alpha.npy")
        self.beta.numpy().tofile(f"model_fit/%s" % (class_) + "/beta.npy")
        self.gmma.numpy().tofile(f"model_fit/%s" % (class_) + "/gmma.npy")
        self.delta.numpy().tofile(f"model_fit/%s" % (class_) + "/delta.npy")

    def load_model(self, class_):
        self.alpha.assign(np.fromfile(f"model_fit/%s" % (class_) + "/alpha.npy", dtype=np.float32).item())
        self.beta.assign(np.fromfile(f"model_fit/%s" % (class_) + "/beta.npy", dtype=np.float32).item())
        self.gmma.assign(np.fromfile(f"model_fit/%s" % (class_) + "/gmma.npy", dtype=np.float32).item())
        self.delta.assign(np.fromfile(f"model_fit/%s" % (class_) + "/delta.npy", dtype=np.float32).item())

    def update_parameters(self, actual_value, val_X):
        previous_loss = []
        start = time.time()
        while True:
            with tf.GradientTape() as tape:
                predicted_value = self.adaptive_forecasting(val_X)
                loss = tf.reduce_mean(tf.square(actual_value - predicted_value))
            gradients = tape.gradient(loss, [self.alpha, self.beta, self.gmma, self.delta])
            self.optimizer.apply_gradients(zip(gradients, [self.alpha, self.beta, self.gmma, self.delta]))
            previous_loss.append(loss)
            end = time.time()
            if self.sliding_window_loss_threshold(previous_loss, self.slide_window,
                                                  threshold=self.threshold) or end - start > 3000:
                break

    def train(self, trainX, trainY, window_size):
        for i in range(0, len(trainX) - window_size, window_size):
            subset = trainX[i:i + window_size]
            val_y = trainY[i:i + window_size]
            self.L = self.L_m.predict(subset.reshape(-1, 1, 3)).reshape(-1)
            self.update_parameters(val_y.reshape(-1), subset)

    def online_predict(self, train_x, testX, testY):
        dynamic = Dynamic(k=self.k)  
        i = 0
        preds_list = []
        while i < len(testX) - self.k + 1:
            val_X = testX[i: i + self.k, :]
            subset, indices = dynamic.find_similar_samples(val_X, train_x)
            subset_Y = testY[i: i + self.k]
            self.L = self.L_m.predict(subset.reshape(-1, 1, 3)).reshape(-1)
            preds = self.adaptive_forecasting(subset)
            preds_list.append(preds)
            self.update_parameters(subset_Y.reshape(-1), subset)
            i += self.k
        return np.array(preds_list).reshape(-1, 1)
