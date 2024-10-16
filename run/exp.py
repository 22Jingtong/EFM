import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model.EFM import Expert_PLUS_EnsembleModel
from util.data_process import Outliers_data, density, to_kmeans_wspd, prepare_data_
from util.error import Error


def exp_EFM(args):
    with tf.device("/GPU:0"):
        df = pd.read_csv(args.filepath, header=0, encoding='utf-8', delimiter=',', na_values=['NaN'])
        Outlied = Outliers_data(df)
        Outlied['date'] = pd.Series(Outlied['date']).astype('datetime64[ns]')
        for id in range(1, 135, 5):
            data_ID = Outlied[Outlied["TurbID"].isin([id])]
            df_density = density(data_ID)
            df_cluster = to_kmeans_wspd(df_density, args.clustering_num)
            # df_cluster = to_kmeans_patv(df_density, args.clustering_num)
            df_cluster.set_index("date", inplace=True)
            for i in range(0, args.clustering_num):
                data_prepare = prepare_data_(df_cluster, args.window_size, i)
                train_X = data_prepare[2]
                trainY = data_prepare[3]
                test_X = data_prepare[4]
                test_y = data_prepare[5]
                scale_new = MinMaxScaler()
                scale_new.min_, scale_new.scale_ = data_prepare[0].min_[0], data_prepare[0].scale_[0]
                ensemble_model = Expert_PLUS_EnsembleModel(args.slide_window, args.dynamic_window, args.n_vars, i)
                ensemble_model.train(train_X, trainY, args.window_size)
                ensemble_model.save_model(i)

                ensemble_model.load_model(i)
                yhat = ensemble_model.online_predict(train_X, test_X, test_y)
                error = Error(scale_new, yhat, test_y)

                rmse = error.get_rmse()
                mae = error.get_mae()
                mape = error.get_mape()
                print(f"ID: {id}, Cluster: {i}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")
