from exp.exp_basic import Exp_Basic
from models import ReTATSF
import torch
from torch import nn
from torch import optim
from data_provider.data_factory import ReTATSF_weather_data_provider
import time
import numpy as np
import os
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from torch.optim import lr_scheduler
from utils.metrics import metric

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = ReTATSF.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag, target_ids):
        data_set, data_loader = ReTATSF_weather_data_provider(self.args, flag, target_ids, self.device)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
                    batch_TS_database, batch_qt, batch_newsdatabase) in enumerate(vali_loader):
                batch_target_series_x = batch_target_series_x.float().to(self.device)
                batch_target_series_y = batch_target_series_y.float().to(self.device)
                batch_TS_database = batch_TS_database.float().to(self.device)
                batch_qt = batch_qt.float().to(self.device)
                batch_newsdatabase = batch_newsdatabase.float().to(self.device)

                time_now = time.time()

                outputs = self.model(batch_target_series_x, batch_TS_database, batch_qt, batch_newsdatabase)

                total_time += time.time() - time_now

                pred = outputs.detach().cpu()
                true = batch_target_series_y.detach().cpu()

                loss = criterion(pred, true)

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

        #target_time = time.time()
        #for target_id in self.target_ids:
            #print('training Target: ', target_id)

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
                    batch_TS_database, batch_qt, batch_newsdatabase) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_target_series_x = batch_target_series_x.float().to(self.device)
                batch_target_series_y = batch_target_series_y.float().to(self.device)

                batch_TS_database = batch_TS_database.float().to(self.device)
                batch_qt = batch_qt.float().to(self.device)
                batch_newsdatabase = batch_newsdatabase.float().to(self.device)
                # print('batch_target_series_x: ', batch_target_series_x)
                # print('batch_target_series_y: ', batch_target_series_y)

                outputs = self.model(batch_target_series_x, batch_TS_database, batch_qt, batch_newsdatabase)

                loss = criterion(outputs, batch_target_series_y)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - iterstage_time) / iter_count
                    left_time_this_target = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time this target: {:.4f}s'.format(speed, left_time_this_target))
                    iter_count = 0
                    iterstage_time = time.time()

                loss.backward()
                model_optim.step()

                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

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

            # print("Target {0} training cost time {1} Last Epoch: Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #             target_id, time.time()-target_time, train_loss, vali_loss, test_loss))
            #time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            #best_target_model_path = path + '/' + target_id + '_best_checkpoint.pth'
            #torch.save(self.model.state_dict(), best_target_model_path)

        return self.model

    def test(self, setting, test=0):
        print(f"Test for {setting}")
        setting = time.strftime("%m%d_%H%M%S_", time.localtime()) + setting
        folder_path = './M_test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.target_ids = self.args.target_ids
        #for target_id in self.target_ids:
        test_data, test_loader = self._get_data(flag='test', target_ids=self.target_ids)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                #os.path.join(self.args.checkpoints, "0311_223330_" + setting + "/p (mbar)_best_checkpoint.pth")
                os.path.join(self.args.checkpoints, "0325_185930_Target_['p (mbar)', 'T (degC)', 'Tpot (K)'] SeqLen_60 PredLen_14 Train_1 GPU_True Kt_5 Kn6 Naggregation_3 Nperseg_30 LR_0.0001 Itr_1 bs_64/p (mbar)T (degC)Tpot (K)_best_checkpoint.pth")
            ))

        if self.args.test_flop:
            test_params_flop(self.model, test_loader, self.device)
            exit()

        preds = []
        trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_target_series_x, batch_target_series_y,
                    batch_TS_database, batch_qt, batch_newsdatabase) in enumerate(test_loader):
                batch_target_series_x = batch_target_series_x.float().to(self.device)
                batch_target_series_y = batch_target_series_y.float().to(self.device)

                batch_TS_database = batch_TS_database.float().to(self.device)
                batch_qt = batch_qt.float().to(self.device)
                batch_newsdatabase = batch_newsdatabase.float().to(self.device)

                outputs = self.model(batch_target_series_x, batch_TS_database, batch_qt, batch_newsdatabase)

                    # outputs = outputs.permute(0, 2, 1)#[b, 1, l]->[b, l, 1]
                    # batch_target_series_x = batch_target_series_x.permute(0, 2, 1)#[b, 1, l]->[b, l, 1]
                    # batch_target_series_x = batch_target_series_x.permute(0, 2, 1)#[b, 1, l]->[b, l, 1]
                    #
                    # b, l, c = outputs.shape
                    # outputs = outputs.reshape(-1, c)
                    # batch_target_series_y = batch_target_series_y.reshape(-1, c)
                    #
                    # outputs = outputs.detach().cpu().numpy()
                    # batch_target_series_y = batch_target_series_y.detach().cpu().numpy()
                    # outputs = test_data.scaler.inverse_transform(outputs).to(self.device)
                    # batch_target_series_y = test_data.scaler.inverse_transform(batch_target_series_y).to(self.device)
                    #
                    # outputs = outputs.reshape(b, l, c)
                    # batch_target_series_y = batch_target_series_y.reshape(b, l, c)
                    #
                    # b, l, c = batch_target_series_x.shape
                    # batch_target_series_x = batch_target_series_x.reshape(-1, c)
                    # batch_target_series_x = batch_target_series_x.detach().cpu().numpy()
                    # batch_target_series_x = test_data.scaler.inverse_transform(batch_target_series_x).to(self.device)
                    #
                    # #outputs = outputs.squeeze(-1)#[b, l]
                    # batch_target_series_x = batch_target_series_x.permute(0, 2, 1)  # [b, l, 1]->[b, 1, l]
                    # batch_target_series_x = batch_target_series_x.permute(0, 2, 1)  # [b, l, 1]->[b, 1, l]
                    # batch_target_series_y = batch_target_series_y.permute(0, 2, 1)  # [b, l, 1]->[b, 1, l]

                outputs = outputs.detach().cpu().numpy()
                batch_target_series_x = batch_target_series_x.detach().cpu().numpy()
                batch_target_series_y = batch_target_series_y.detach().cpu().numpy()

                pred = outputs
                true = batch_target_series_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_target_series_x)
                if i % 20 == 0:
                    j=0
                    for target_id in self.target_ids:
                        input = batch_target_series_x
                        gt = np.concatenate((input[0, j, :], true[0, j, :]), axis=0)
                        pd = np.concatenate((input[0, j, :], pred[0, j, :]), axis=0)
                        #time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        # print("gt: ", gt)
                        # print("pd: ", pd)
                        # print('j: ', j)
                        visual(gt, pd, os.path.join(folder_path, target_id+'_'+str(i)+'.pdf'))
                        j+=1

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('targets_{}: mse_{}, mae_{}'.format("".join(self.target_ids), mse, mae))

        info_save_dir = os.path.join(folder_path, 'result.txt')
        f = open(info_save_dir, 'a')
        #f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(os.path.join(folder_path, "".join(self.target_ids)+'_pred.npy'), preds)
        np.save(os.path.join(folder_path, "".join(self.target_ids)+'_trues.npy'), trues)
        np.save(os.path.join(folder_path, "".join(self.target_ids)+'_x.npy'), inputx)