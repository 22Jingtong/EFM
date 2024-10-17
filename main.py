import argparse
import warnings
from run.exp import exp_EFM

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Tensorflow wind prediction model - EFM+')
    #model
    parser.add_argument('--n_batch', type=int, default=12)
    parser.add_argument('--n_vars', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=288)
    parser.add_argument('--drop_out', type=int, default=0.1)
    parser.add_argument('--Turbine', type=int, default=135)
    parser.add_argument('--filepath', type=str, default=r'./data/all_wind_stations_production.csv')
    parser.add_argument('--clustering_num', type=int, default=5)
    parser.add_argument('--slide_window', type=int, default=50)
    parser.add_argument('--dynamic_window', type=int, default=12)
    parser.add_argument('--lvbo', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    exp_EFM(args)