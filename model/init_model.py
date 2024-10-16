from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.layers import Dense, LSTM, Input, Dropout
from keras.layers import Layer
from keras.models import Model
from tensorflow.python.keras.layers import Dropout
import keras.backend as K
from tensorflow.python.keras import regularizers
import tensorflow as tf


class Attention(Layer):
    def __init__(self, step_dim, bias=True, **kwargs):
        self.b = None
        self.W = None
        self.step_dim = step_dim
        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer='uniform',
                                 trainable=True)
        if self.bias: self.b = self.add_weight(name='{}_b'.format(self.name),
                                               shape=(input_shape[1],),
                                               initializer='uniform',
                                               trainable=True)
        super(Attention, self).build(input_shape)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'step_dim': self.step_dim,
            'bias': self.bias,
        })
        return config

    def call(self, x, mask=None, **kwargs):

        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def lstm_m(window_size, n_vars, n_neurons=50, dropout=0.05, optimizer='adam'):
    inputs = Input(shape=(window_size, n_vars))
    # 使用双向LSTM
    lstm_hide = LSTM(units=int(n_neurons), return_sequences=True)(inputs)
    lstm_hide1 = LSTM(units=int(n_neurons), return_sequences=True)(lstm_hide)
    lstm_hide_drop1 = Dropout(dropout)(lstm_hide1)
    lstm_hide2 = LSTM(units=int(n_neurons), return_sequences=True)(lstm_hide_drop1)
    lstm_hide_drop3 = Dropout(dropout)(lstm_hide2)
    lstm_out = LSTM(units=int(n_neurons), return_sequences=True)(lstm_hide_drop3)
    attention_out = Attention(step_dim=window_size)(lstm_out)
    output = Dense(units=window_size, kernel_regularizer=regularizers.l2(0.01))(attention_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss="mse", optimizer=optimizer, run_eagerly=True)
    return model


class LSTM_model:
    def __init__(self, window_size, n_vars, n_neurons=50, dropout=0.05, class_wspd=0):
        self.n_vars = n_vars
        self.window_size = window_size
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.optimizer = tf.keras.optimizers.SGD(lr=1e-3)
        self.class_wspd = class_wspd
        self._buildModel_()

    def _buildModel_(self):
        inputs = Input(shape=(self.window_size, self.n_vars))
        lstm_hide = LSTM(units=int(self.n_neurons), return_sequences=True)(inputs)
        lstm_hide1 = LSTM(units=int(self.n_neurons), return_sequences=True)(lstm_hide)
        lstm_hide_drop1 = Dropout(self.dropout)(lstm_hide1)
        lstm_hide2 = LSTM(units=int(self.n_neurons), return_sequences=True)(lstm_hide_drop1)
        lstm_hide_drop3 = Dropout(self.dropout)(lstm_hide2)
        lstm_out = LSTM(units=int(self.n_neurons), return_sequences=True)(lstm_hide_drop3)
        attention_out = Attention(step_dim=self.window_size)(lstm_out)
        output = Dense(units=self.window_size, kernel_regularizer=regularizers.l2(0.01))(attention_out)
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(loss="mse", optimizer=self.optimizer, run_eagerly=True)
        return

    def fit(self, x, y):
        x = x.reshape((-1, 1, self.n_vars))
        trian_x = x[:int(len(x) * 0.8)]
        trian_y = y[:int(len(y) * 0.8)]
        val_x = x[int(len(x) * 0.8):]
        val_y = y[int(len(y) * 0.8):]
        train_dataset = tf.data.Dataset.from_tensor_slices((trian_x, trian_y))
        train_dataset = train_dataset.shuffle(60000).batch(64)
        evaluate_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        evaluate_dataset = evaluate_dataset.batch(64)
        self.model.fit(train_dataset, epochs=100)
        self.model.evaluate(evaluate_dataset)
        self.model.save('../model_fit/%s' % (self.class_wspd) + '/lstm_model.h5')

    @tf.function
    def predict(self, x):
        x = x.reshape((-1, 1, self.n_vars))
        return self.model.predict(x).reshape(-1, 1)

    def evaluate(self, x, y):
        x = x.reshape((-1, 1, self.n_vars))
        evaluate_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        evaluate_dataset = evaluate_dataset.batch(64)
        return self.model.evaluate(evaluate_dataset)


class machine_model:
    def __init__(self):
        self.xgb = self.__buildXgbModel__()
        self.rfr = self.__buildRfrModel__()

    @staticmethod
    def __buildXgbModel__():
        xgb = XGBRegressor(tree_method="gpu_hist", gpu_id=0, colsample_bytree=0.8, learning_rate=0.01, max_depth=3,
                                min_child_samples=30, n_estimators=300, reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, random_state=42)
        return xgb

    @staticmethod
    def __buildRfrModel__():
        rfr = RandomForestRegressor(n_estimators=700, max_features='log2', random_state=42, max_depth=17,
                                    bootstrap=True, min_samples_leaf=7, min_samples_split=11)
        return rfr

    def fit(self, x, y):
        self.xgb.fit(x, y)
        self.rfr.fit(x, y)

    def predict(self, x):
        xgb_pred = self.xgb.predict(x).reshape(-1, 1)
        rfr_pred = self.rfr.predict(x).reshape(-1, 1)
        return xgb_pred, rfr_pred


