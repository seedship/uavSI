import LateralDynamics as lls
import DataProcessing as dp

import pandas as pd

if __name__ == '__main__':
    resp_2l_2r = pd.read_csv('~/uavSI/data/lat/csvs/ail_resp_2L_2R.csv')
    resp_2l_2r = dp.fromAlvolo2019(resp_2l_2r)
    # resp_2l_2r = lls.TrimData(resp_2l_2r, lls.LinearizationPoint())
    C_l_d_a, C_n_d_a = lls.aileron(resp_2l_2r)
    print('ail_resp_2L_2R.csv')
    print('C_l_d_a:', C_l_d_a)
    print('C_n_d_a:', C_n_d_a)

    resp_l = pd.read_csv('~/uavSI/data/lat/csvs/ail_resp_L.csv')
    resp_l = dp.fromAlvolo2019(resp_l)
    # resp_l = lls.TrimData(resp_l, lls.LinearizationPoint())
    C_l_d_a, C_n_d_a = lls.aileron(resp_l)
    print('ail_resp_L.csv')
    print('C_l_d_a:', C_l_d_a)
    print('C_n_d_a:', C_n_d_a)

    resp_l_r = pd.read_csv('~/uavSI/data/lat/csvs/ail_resp_L_R.csv')
    resp_l_r = dp.fromAlvolo2019(resp_l_r)
    # resp_l_r = lls.TrimData(resp_l_r, lls.LinearizationPoint())
    C_l_d_a, C_n_d_a = lls.aileron(resp_l_r)
    print('ail_resp_L_R.csv')
    print('C_l_d_a:', C_l_d_a)
    print('C_n_d_a:', C_n_d_a)

    resp_r = pd.read_csv('~/uavSI/data/lat/csvs/ail_resp_R.csv')
    resp_r = dp.fromAlvolo2019(resp_r)
    # resp_r = lls.TrimData(resp_r, lls.LinearizationPoint())
    C_l_d_a, C_n_d_a = lls.aileron(resp_r)
    print('ail_resp_R.csv')
    print('C_l_d_a:', C_l_d_a)
    print('C_n_d_a:', C_n_d_a)

    resp_r_l = pd.read_csv('~/uavSI/data/lat/csvs/ail_resp_R_L.csv')
    resp_r_l = dp.fromAlvolo2019(resp_r_l)
    # resp_r_l = lls.TrimData(resp_r_l, lls.LinearizationPoint())
    C_l_d_a, C_n_d_a = lls.aileron(resp_r_l)
    print('ail_resp_R_L.csv')
    print('C_l_d_a:', C_l_d_a)
    print('C_n_d_a:', C_n_d_a)