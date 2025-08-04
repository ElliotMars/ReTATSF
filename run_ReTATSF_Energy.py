import argparse
import torch
from exp.exp_ReTATSF_Energy import Exp_Main
import random
import numpy as np
import multiprocessing

parser= argparse.ArgumentParser(description='Retrieval Based Text Augmented Time Series Forecasting')

#random seed
parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

#basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

#ReTATSF dataloader
parser.add_argument('--root_path', type=str, default='./dataset', help='root path')
parser.add_argument('--TS_data_path', type=str, default='Time-MMD/numerical/Energy/Energy.parquet', help='Time series data path')
parser.add_argument('--QT_emb_path', type=str, default='Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2-ForecastingPoint', help='Query text embedding data path')
parser.add_argument('--Des_emb_path', type=str, default='Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2-Description', help='Description embedding data path')
parser.add_argument('--NewsDatabase_path', type=str, default='Time-MMD/textual/Energy/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2', help='News database path')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='model checkpoints path')
parser.add_argument('--num_data', type=int, default=1, help='number of data points in total')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

#ReTATSF
#Coherence Analysis
#parser.add_argument('--freq_sample', type=float, default=0.00167, help='frequency of sampling')#10min
parser.add_argument('--nperseg', type=int, default=30, help='number of samples per segment')#seq_len//2
parser.add_argument('--nref', type=int, default=5, help='number of reference TS')#nref < 20
#Content Synthesis
parser.add_argument('--naggregation', type=int, default=3, help='number of aggregation module')
#text retrival
parser.add_argument('--nref_text', type=int, default=6, help='number of text retrival')#nref_text=nref+1
parser.add_argument('--qt_ref_text', type=int, default=5, help='number of text retrival by qt')

#forecasting task
parser.add_argument('--seq_len', type=int, default=60, help='sequence length')
parser.add_argument('--pred_len', type=int, default=14, help='predicted length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--target_ids', nargs='+', required=True, type=str, help='names of target TS')

#optimization
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='l2 lambda')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_multi_gpu:
        multiprocessing.set_start_method("spawn")
        # torch.multiprocessing.set_start_method('spawn', force=True)

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            #setting record of experiments
            setting = 'Target_{} SeqLen_{} PredLen_{} Train_{} GPU_{} Kt_{} Kn{} Naggregation_{} Nperseg_{} LR_{} Itr_{} bs_{}'.format(
                args.target_ids[0]+'-'+args.target_ids[-1],
                args.seq_len,
                args.pred_len,
                args.is_training,
                args.use_gpu,
                args.nref,
                args.nref_text,
                args.naggregation,
                args.nperseg,
                args.learning_rate,
                args.itr,
                args.batch_size
            )

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = 'Target_{} SeqLen_{} PredLen_{} Train_{} GPU_{} Kt_{} Kn{} Naggregation_{} Nperseg_{} LR_{} Itr_{}'.format(
                args.target_ids[0]+'-'+args.target_ids[-1],
                args.seq_len,
                args.pred_len,
                args.is_training,
                args.use_gpu,
                args.nref,
                args.nref_text,
                args.naggregation,
                args.nperseg,
                args.learning_rate,
                args.itr
            )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()