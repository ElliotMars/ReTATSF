from exp.exp_basic import Exp_Basic
from models import ReTATSF
import torch
from torch import nn
from torch import optim
from data_provider.data_factory import ReTATSF_Energy_data_provider
import time
import numpy as np
import os
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from torch.optim import lr_scheduler
from utils.metrics import metric
import glob
import re

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = ReTATSF.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag, target_ids):
        data_set, data_loader = ReTATSF_Energy_data_provider(self.args, flag, target_ids)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_target_series_x, batch_target_series_y,
                    batch_TS_database, batch_qt, batch_des, batch_newsdatabase) in enumerate(vali_loader):
                batch_target_series_x = batch_target_series_x.float().to(self.device)
                batch_target_series_y = batch_target_series_y.float().to(self.device)
                batch_TS_database = batch_TS_database.float().to(self.device)
                batch_qt = batch_qt.float().to(self.device)
                batch_des = batch_des.float().to(self.device)
                batch_newsdatabase = batch_newsdatabase.float().to(self.device)

                time_now = time.time()

                outputs, _, _ = self.model(batch_target_series_x, batch_TS_database, batch_qt, batch_des, batch_newsdatabase)

                total_time += time.time() - time_now

                pred = outputs.detach().cpu()
                true = batch_target_series_y.detach().cpu()

                if i % 5 == 0:
                    j=0
                    for target_id in self.target_ids:
                        input = batch_target_series_x.detach().cpu()
                        gt = np.concatenate((input[0, j, :], true[0, j, :]), axis=0)
                        pd = np.concatenate((input[0, j, :], pred[0, j, :]), axis=0)
                        visual(gt, pd, input.shape[-1], os.path.join('./fig/vali_results', target_id+'_'+str(i)+'.pdf'))
                        j+=1

                label_len = self.args.pred_len if self.args.pred_len < self.args.label_len else self.args.label_len
                loss = criterion(pred[:, :, :label_len], true[:, :, :label_len])

                total_loss.append(loss)

            print('vali time: ', total_time/(i+1))
            total_loss = np.average(total_loss)
            self.model.train()
            return total_loss

    def train(self, setting):
        # take the current time as MMDD_HHMMSS
        setting = time.strftime("%m%d_%H%M%S_", time.localtime()) + setting

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self.target_ids = self.args.target_ids

        train_data, train_loader = self._get_data(flag='train', target_ids=self.target_ids)
        vali_data, vali_loader = self._get_data(flag='val', target_ids=self.target_ids)
        test_data, test_loader = self._get_data(flag='test', target_ids=self.target_ids)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        epoch_time = time.time()
        for epoch in range(self.args.train_epochs):

            train_loss = []

            iter_count = 0
            self.model.train()

            iterstage_time = time.time()
            for i, (batch_target_series_x, batch_target_series_y,
                    batch_TS_database, batch_qt, batch_des, batch_newsdatabase) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_target_series_x = batch_target_series_x.float().to(self.device)
                batch_target_series_y = batch_target_series_y.float().to(self.device)

                batch_TS_database = batch_TS_database.float().to(self.device)
                batch_qt = batch_qt.float().to(self.device)
                batch_des = batch_des.float().to(self.device)
                batch_newsdatabase = batch_newsdatabase.float().to(self.device)

                outputs, temp_emb, text_emb = self.model(batch_target_series_x, batch_TS_database, batch_qt, batch_des, batch_newsdatabase)

                label_len = self.args.pred_len if self.args.pred_len < self.args.label_len else self.args.label_len
                loss = criterion(outputs[:, :, :label_len], batch_target_series_y[:, :, :label_len])
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - iterstage_time) / iter_count
                    left_time_this_target = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time : {:.4f}s'.format(speed, left_time_this_target))
                    iter_count = 0
                    iterstage_time = time.time()

                    import matplotlib.pyplot as plt
                    import torch.nn.init as init

                    # 设置样本和通道索引
                    B_idx = 0
                    C_idx = 0

                    # 获取原始张量 [L, D] 和 [L+H, D]
                    sample_channel_temp = temp_emb[B_idx, C_idx]  # shape: [L, D]
                    sample_channel_text = text_emb[B_idx, C_idx]  # shape: [L+H, D]

                    # 定义线性投影网络（共享或不共享都行，这里用两个）
                    proj = nn.Linear(sample_channel_temp.shape[-1], 1).to(sample_channel_temp.device)
                    # 初始化所有权重为相同的值，例如 1.0
                    init.constant_(proj.weight, 1.0)  # 所有权重设为 1.0
                    init.constant_(proj.bias, 0.0)  # 偏置设为 0.0（或也可设为 1.0）

                    # 线性映射到一维
                    temp_proj_1d = proj(sample_channel_temp).squeeze(-1).detach().cpu().numpy()  # shape: [L]
                    text_proj_1d = proj(sample_channel_text).squeeze(-1).detach().cpu().numpy()  # shape: [L+H]

                    # 画图
                    plt.figure(figsize=(12, 5))
                    plt.plot(text_proj_1d, label='text_emb 1D Projection', linewidth=2)
                    plt.plot(temp_proj_1d, label='temp_emb 1D Projection', linewidth=2)
                    plt.xlabel('Time Step')
                    plt.ylabel('Projected Value')
                    plt.title('1D Projection of temp_emb and text_emb')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig('/data/dyl/ReTATSF/fig/projection.pdf')
                    plt.close()

                loss.backward()
                model_optim.step()

                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                #scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, self.target_ids)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        return self.model

    def test(self, setting, test=0):
        print(f"Test for {setting}")
        setting = time.strftime("%m%d_%H%M%S_", time.localtime()) + setting
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.target_ids = self.args.target_ids
        #for target_id in self.target_ids:
        test_data, test_loader = self._get_data(flag='test', target_ids=self.target_ids)



        self.model = self.model.module if hasattr(self.model, 'module') else self.model
        self.load_latest_checkpoint(setting)
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)

        if self.args.test_flop:
            test_params_flop(self.model, test_loader, self.device)
            exit()

        preds = []
        trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_target_series_x, batch_target_series_y,
                    batch_TS_database, batch_qt, batch_des, batch_newsdatabase) in enumerate(test_loader):
                batch_target_series_x = batch_target_series_x.float().to(self.device)
                batch_target_series_y = batch_target_series_y.float().to(self.device)

                batch_TS_database = batch_TS_database.float().to(self.device)
                batch_qt = batch_qt.float().to(self.device)
                batch_des = batch_des.float().to(self.device)
                batch_newsdatabase = batch_newsdatabase.float().to(self.device)

                outputs, _, _ = self.model(batch_target_series_x, batch_TS_database, batch_qt, batch_des, batch_newsdatabase)

                outputs = outputs.detach().cpu().numpy()
                batch_target_series_x = batch_target_series_x.detach().cpu().numpy()
                batch_target_series_y = batch_target_series_y.detach().cpu().numpy()

                pred = outputs
                true = batch_target_series_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_target_series_x)
                if i % 5 == 0:
                    j=0
                    for target_id in self.target_ids:
                        input = batch_target_series_x
                        gt = np.concatenate((input[0, j, :], true[0, j, :]), axis=0)
                        pd = np.concatenate((input[0, j, :], pred[0, j, :]), axis=0)
                        visual(gt, pd, input.shape[-1], os.path.join(folder_path, target_id+'_'+str(i)+'.pdf'))
                        j+=1

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        label_len = self.args.pred_len if self.args.pred_len < self.args.label_len else self.args.label_len
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds[:, :, :label_len], trues[:, :, :label_len])
        print('targets_{}: mse_{}, mae_{}'.format("".join(self.target_ids), mse, mae))

        info_save_dir = os.path.join(folder_path, 'result.txt')
        f = open(info_save_dir, 'a')
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(os.path.join(folder_path, "".join(self.target_ids)+'_pred.npy'), preds)
        np.save(os.path.join(folder_path, "".join(self.target_ids)+'_trues.npy'), trues)
        np.save(os.path.join(folder_path, "".join(self.target_ids)+'_x.npy'), inputx)

    def load_latest_checkpoint(self, setting):
        setting = setting[12:]
        checkpoint_root = self.args.checkpoints

        # 忽略 setting 中的 Train_0 或 Train_1 差异
        generalized_setting = re.sub(r'Train_[01]', 'Train_', setting)

        # 匹配所有目录名中含有 Train_ 的，且其余部分与 setting 匹配的目录
        all_dirs = glob.glob(os.path.join(checkpoint_root, "*"))
        matched_dirs = [
            d for d in all_dirs
            if os.path.isdir(d) and generalized_setting in re.sub(r'Train_[01]', 'Train_', os.path.basename(d))
        ]

        if not matched_dirs:
            raise FileNotFoundError(
                f"No checkpoint directory matching generalized pattern '{generalized_setting}' found in {checkpoint_root}")

        # 按最近修改时间排序
        matched_dirs.sort(key=os.path.getmtime, reverse=True)
        latest_dir = matched_dirs[0]

        # 获取该目录下的 _best_checkpoint.pth 文件
        model_files = glob.glob(os.path.join(latest_dir, "*_best_checkpoint.pth"))
        if not model_files:
            raise FileNotFoundError(f"No '_best_checkpoint.pth' file found in {latest_dir}")

        best_model_path = model_files[0]
        print(f"正在加载模型: {best_model_path}")

        self.model.load_state_dict(torch.load(best_model_path))