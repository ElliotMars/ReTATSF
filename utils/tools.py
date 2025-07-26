import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch import nn
from thop import profile
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, target_ids):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, target_ids)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, target_ids)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, target_ids):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        id_str = ""
        for target_id in target_ids:
            id_str += target_id
        #time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #torch.save(model.state_dict(), path + '/' + id_str + '_best_checkpoint.pth')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), path + '/' + id_str + '_best_checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + id_str + '_best_checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

def visual(true, preds=None, seq_len=60, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    #draw a vertical line at x = 60 with grey dashed line style linewidth=2
    plt.axvline(x=seq_len, linestyle='--', color='grey', linewidth=3)
    plt.axvline(x=seq_len+15, linestyle='--', color='grey', linewidth=2)
    plt.axvline(x=seq_len-5, linestyle='--', color='grey', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)

    plt.legend()
    if name != "":
        plt.savefig(name, bbox_inches='tight')
    else:
        plt.show()

# def test_params_flop(model, batch_target_series_x, batch_TS_database, batch_qt, batch_newsdatabase):
#     """
#     If you want to thest former's flop, you need to give default value to inputs in model.forward(),
#     the following code can only pass one argument to forward()
#     """
#     model_params = sum(p.numel() for p in model.parameters())
#     print(f'INFO: Trainable parameter count: {model_params / 1e6:.2f}M')
#
#     # 确保 batch 维度去掉，只传入输入张量的形状
#     batch_target_series_x_shape = batch_target_series_x.shape[1:]  # (seq_len, feature_dim)
#     batch_TS_database_shape = batch_TS_database.shape[1:]
#     batch_qt_shape = batch_qt.shape[1:]
#     batch_newsdatabase_shape = batch_newsdatabase.shape[1:]
#
#     def input_constructor(input_res):
#         """构造 `forward()` 需要的所有输入"""
#         return (torch.randn(1, *batch_target_series_x_shape).cuda(),
#                 torch.randn(1, *batch_TS_database_shape).cuda(),
#                 torch.randn(1, *batch_qt_shape).cuda(),
#                 torch.randn(1, *batch_newsdatabase_shape).cuda())
#
#     from ptflops import get_model_complexity_info
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(
#             model.cuda(),
#             batch_target_series_x_shape,
#             as_strings=True,
#             print_per_layer_stat=True,
#             input_constructor=input_constructor  # 关键：构造多输入
#         )
#     print(f'Computational complexity: {macs}')
#     print(f'Number of parameters: {params}')

def test_params_flop(model, test_loader, device):
    batch_target_series_x, batch_target_series_y, batch_TS_database, batch_qt, batch_newsdatabase = next(iter(test_loader))
    batch_target_series_x = batch_target_series_x.float().to(device)
    batch_target_series_y = batch_target_series_y.float().to(device)
    batch_TS_database = batch_TS_database.float().to(device)
    batch_qt = batch_qt.float().to(device)
    batch_newsdatabase = batch_newsdatabase.float().to(device)
    time_now = time.time()

    # for param in self.model.parameters():
    #     print(param.device)
    model_to_profile = model.module if isinstance(model, (
    torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model

    macs, params = profile(model_to_profile, inputs=(batch_target_series_x, batch_TS_database, batch_qt, batch_newsdatabase))
    total_time = time.time() - time_now
    print('FLOPs: ', macs)
    print('params: ', params)
    print('Total time: ', total_time)