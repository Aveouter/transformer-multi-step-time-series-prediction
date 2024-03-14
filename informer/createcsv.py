import numpy as np
import pandas as pd
# 指定.npy文件路径
file_path1 = "results/informer_custom_ftMS_sl120_ll120_pl60_dm512_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/true.npy"
file_path2 = "results/informer_custom_ftMS_sl120_ll120_pl60_dm512_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/pred.npy"
file_path3 = "results/informer_custom_ftMS_sl120_ll120_pl60_dm512_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/real_prediction.npy"
path_of_data = "final_data-CH41.csv"
 # results\informer_custom_ftM_sl120_ll120_pl60_dm512_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1
#W:\ProjectCAS\TemperaturePredictionOfBattery\results\informer_custom_ftMS_sl120_ll120_pl60_dm512_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0
#W:\ProjectCAS\TemperaturePredictionOfBattery\results\informer_custom_ftMS_sl120_ll120_pl60_dm512_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1\real_prediction.npy
# 使用NumPy加载.npy文件
true_value = []
pred_value = []
 
# count the mean and std
df_raw = pd.read_csv(path_of_data)
target = df_raw['Temperature']
mean = target.mean(0)
std = target.std(0)
# print(mean,std)

data1 = np.load(file_path1)
data2 = np.load(file_path2)
real_prediction = np.load(file_path3)

data1 = (data1 * std) + mean
data2 = (data2 * std) + mean
real_prediction = (real_prediction * std) + mean
print(f'realp{real_prediction.shape}')


import matplotlib.pyplot as plt
plt.figure()
plt.plot(data1[0,:,-1], label='Prediction')
plt.plot(data2[0,:,-1], label='true')
plt.legend()
plt.show()

# print(data2)
# for i in range(24):
#     true_value.append(data2[0][i][3])
#     pred_value.append(data1[0][i][3])
 
# # 打印内容
# print(true_value[:5])
# print(pred_value[:5])
 
# df = pd.DataFrame({'real': true_value, 'pred': pred_value})
 
# df.to_csv('results.csv', index=False)