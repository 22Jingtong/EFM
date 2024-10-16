import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

def Outliers_data(df):
    nan_rows = df.isnull().any(axis=1)
    invalid_cond = (df['Patv'] < 0) | \
                   ((df['Patv'] == 0) & (df['Wspd'] > 2.5)) | \
                   (df['Pab'] > 89) | \
                   ((df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) |
                    (df['Ndir'] > 720))
    df.loc[invalid_cond, df.columns.difference(['TurbID'])] = np.nan
    df['mask'] = np.where(invalid_cond | nan_rows, 0, 1)
    df.loc[df['Patv'] < 0, 'Patv'] = 0
    df['Prtv'] = df['Prtv'].abs()
    df = df.groupby('TurbID').apply(lambda x: x.interpolate().ffill().bfill().fillna(0))
    return df


def density(df):
    wspd = df["Wspd"].values
    prtv = df["Prtv"].values
    data = np.vstack([wspd, prtv])
    kde = gaussian_kde(data)
    density = kde(data)
    # # 设置密度阈值
    density_threshold = 0.000008  # 根据实际情况设置阈值

    # 判断散点是否在不规则区域内
    in_irregular_region = density > density_threshold
    label = np.zeros(prtv.shape)
    # 输出结果
    for i, point in enumerate(zip(wspd, prtv)):
        if in_irregular_region[i]:
            label[i] = 1
        else:
            label[i] = 0
    df = df.assign(label=label)
    return df


def to_kmeans_patv(data):
    Wspd = data["Patv"].values
    k = 5  # 聚类数量
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(Wspd.reshape(-1, 1))
    cluster_labels = kmeans.labels_
    data["level"] = cluster_labels
    return data

def to_kmeans_wspd(data, k=5):
    Wspd = data["Wspd"].values
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(Wspd.reshape(-1, 1))
    cluster_labels = kmeans.labels_
    data["level"] = cluster_labels
    return data



def to_agglomerative(data):
    data = data[data['label'].isin([1])]
    Wspd = data["Wspd"].values
    k = 3
    agg_clustering = AgglomerativeClustering(n_clusters=k)
    agg_clustering.fit(Wspd.reshape(-1, 1))
    labels = agg_clustering.labels_
    data["level"] = labels
    return data


def to_DBSCAN(data):
    data = data[data['label'].isin([1])]
    Wspd = data["Wspd"].values
    k = 5
    dbscan = DBSCAN(eps=0.5, min_samples=k)
    dbscan.fit(Wspd.reshape(-1, 1))

    labels = dbscan.labels_
    data["level"] = labels
    return data


def to_MeanShift(data):
    data = data[data['label'].isin([1])]
    Wspd = data["Wspd"].values

    mean_shift = MeanShift()
    mean_shift.fit(Wspd.reshape(-1, 1))
    labels = mean_shift.labels_
    data["level"] = labels
    return data

def to_GaussianMixture(data):
    data = data[data['label'].isin([1])]
    Wspd = data["Wspd"].values
    gmm = GaussianMixture(n_components=5)  # 5 表示高斯成分的数量
    gmm.fit(Wspd.reshape(-1, 1))

    labels = gmm.predict(Wspd.reshape(-1, 1))
    data["level"] = labels
    return data


def to_SpectralClustering(data):
    data = data[data['label'].isin([1])]
    Wspd = data["Wspd"].values
    spectral_clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors')
    spectral_clustering.fit(Wspd.reshape(-1, 1))
    labels = spectral_clustering.labels_
    data["level"] = labels
    return data

def prepare_data_(df_cluster, window_size=144, class_wspd=0, type_vars=0, lvbo=True):

    dataset = classical_season(df_cluster, class_wspd)
    if type_vars == 0:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Pab", "Prtv"])
    elif type_vars == 1:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Etmp", "Itmp", "Pab", "Prtv"])
    elif type_vars == 2:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Wdir", "Ndir", "Pab", "Prtv"])
    elif type_vars == 3:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Wdir", "Ndir", "Etmp", "Itmp", "Pab", "Prtv"])
    elif type_vars == 4:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Wdir", "Ndir"])
    elif type_vars == 5:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Etmp", "Itmp"])
    else:
        df_patv = dataset.reindex(columns=["Patv", "Wspd", "Wdir", "Ndir", "Etmp", "Itmp", "Pab", "Prtv"])
    values = df_patv.values
    if lvbo:
        poly_order = 1
        slide_window = 36
        # 对每一列属性应用Savitri-Golay滤波器
        smoothed_data = np.zeros_like(values)
        for i in range(values.shape[1]):
            column_data = values[:, i]
            smoothed_column_data = savgol_filter(column_data, slide_window, poly_order)
            smoothed_data[:, i] = smoothed_column_data
    else:
        smoothed_data = values
    # 变量归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(smoothed_data)
    # 分隔数据集，分为训练集和测试集
    train_X, trainY, test_X, test_y = train_test_split(scaled, window_size)
    return scaler, values, train_X, trainY, test_X, test_y

def train_test_split(values, window_size=144):
    n_train = round(window_size)
    start = (len(values) - n_train) % window_size
    train = values[start:-n_train, :]
    test = values[-n_train:, :]

    # 分隔输入X和输出y
    print("train:", train.shape)
    print("test:", test.shape)
    train_X, trainY = train[:, 1:], train[:, :1]
    test_X, test_y = test[:, 1:], test[:, :1]
    return train_X, trainY, test_X, test_y

def classical_season(dataset, class_wspd):
    dataset_mask = dataset[dataset["label"].isin([1]) & dataset["mask"].isin([1])]
    dataset_month = dataset_mask[dataset_mask["level"].isin([class_wspd])]
    dataset_month = dataset_month.drop(['TurbID', 'mask', 'level'], axis=1)
    return dataset_month