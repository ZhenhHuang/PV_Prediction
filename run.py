import argparse
import os

import pandas as pd
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas
from utils import create_logger


fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(
    description='Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str,  default='', help='model id')
parser.add_argument('--model', type=str, default='Myformer',
                    help='model name, options: [DLinear, Informer, FEDformer, Transformer, LSTM]')

# data loader
parser.add_argument('--data', type=str, default='PV', help='dataset type')
parser.add_argument('--root_path', type=str,
                    default='./dataset/PV', help='root path of the data file')
parser.add_argument('--data_path', type=str,
                    default='station00.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='power',
                    help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str,
                    default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96,
                    help='input sequence length')
parser.add_argument('--label_len', type=int, default=48,
                    help='start token length')
parser.add_argument('--pred_len', type=int, default=96,
                    help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=14, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=14, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512,
                    help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2,
                    help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1,
                    help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25,
                    help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str,
                    default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true',
                    help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true',
                    help='whether to predict unseen future data')
parser.add_argument('--conv_size', type=list, default=[4, 6, 2], help='kernel size list for IPConv')
parser.add_argument('--conv_stride', type=list, default=[4, 2, 2], help='stride list for IPConv')
parser.add_argument('--dilation', type=int, default=1, help='dilation for IPConv')
parser.add_argument('--wavelet', type=str, default='db4', help='wavelet name')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# supplementary config for Myformer model
parser.add_argument('--M', type=int, default=32, help='squeezed length')
parser.add_argument('--d_hidden', type=int, default=256, help='dimension of down mlp')

# supplementary config for LSTM model
parser.add_argument('--num_layers', type=int, default=2, help='layers of LSTM')

# optimization
parser.add_argument('--num_workers', type=int, default=0,
                    help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int,
                    default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3,
                    help='early stopping patience')
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1',
                    help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true',
                    help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multile gpus')


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print(args.use_gpu)
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

Exp = Exp_Main

if __name__ == '__main__':
    print(args.data_path)
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path), encoding='gbk')
    input_dims = len(list(df_raw.columns)[1:])
    print(f'enc_in and dec_in: {input_dims}')
    args.enc_in = input_dims
    args.dec_in = input_dims
    del df_raw
    print(f"{args.enc_in}, {args.dec_in}")
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            log_dir = os.path.join('./test_results', setting)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            logger = create_logger(os.path.join(log_dir, 'train.log'))
            logger.info(args)
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print(
                    '>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
