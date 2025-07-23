import numpy as np
trues = np.load("/data/dyl/ReTATSF/M_test_results/0721_221645_Target_Exports-International Trade Balance SeqLen_20 PredLen_40 Train_1 GPU_True Kt_5 Kn6 Naggregation_3 Nperseg_30 LR_0.0001 Itr_1 bs_32/ExportsImportsInternational Trade Balance_trues.npy")
preds = np.load("/data/dyl/ReTATSF/M_test_results/0721_221645_Target_Exports-International Trade Balance SeqLen_20 PredLen_40 Train_1 GPU_True Kt_5 Kn6 Naggregation_3 Nperseg_30 LR_0.0001 Itr_1 bs_32/ExportsImportsInternational Trade Balance_pred.npy")
#inputx = np.load('./M_test_results/0405_002535_Target_p (mbar)-Tlog (degC) SeqLen_60 PredLen_14 Train_1 GPU_True Kt_5 Kn6 Naggregation_3 Nperseg_30 LR_0.0001 Itr_1 bs_128/p (mbar)T (degC)rh (%)VPmax (mbar)wv (m_s)sh (g_kg)Tlog (degC)_x.npy')

print(trues.shape)
print(preds.shape)
#print(inputx.shape)

# 转换形状：
trues_reshaped = trues.transpose(1, 0, 2).reshape(3, -1)
preds_reshaped = preds.transpose(1, 0, 2).reshape(3, -1)

# 计算 MSE（沿最后一个维度）
mse = np.mean((trues_reshaped - preds_reshaped) ** 2, axis=-1)

print("MSE for each of the targets:", mse)