import torch
import pandas as pd
import numpy as np
from utils import *
import argparse
from dataset import *
from SAGAM_Net import *
from datetime import datetime
import time
import logging
import sys
import os
import shutil

#########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.7, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.1, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of prediction')
parser.add_argument('--his_len', type=int, default=12, help='sequence length of historical observation')
parser.add_argument('--channelin', type=int, default=2, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
parser.add_argument("--loss", type=str, default='MaskMAE', help="MAE, MSE, MaskMAE")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--patience", type=float, default=30, help="patience used for early stop")
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--hiddenunits', type=int, default=32, help='number of hidden units')
parser.add_argument('--mem_num', type=int, default=10, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=32, help='dimension of meta-nodes/prototypes')
parser.add_argument('--min', type=float, default=-2, help='Minimum speed error after standardization, -2 for METRLA -5 for BAY')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed.')
opt = parser.parse_args()

model_name = 'SAGAM_Net'
if opt.dataset == 'PEMSBAY':
    opt.min = -5
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'save/{opt.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
shutil.copy2('dataset.py', path)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN_MEM(64,
                    207, # 325
                    input_dim=2,
                    output_dim=12,
                    horizon=12,
                    rnn_units=32*8,
                    steps=12,
                    num_layers=1,
                    cheb_k=3,
                    mem_num=10,
                    mem_dim=32).to(device)
    return model


def print_model(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print('Total parameters:', param_count)


data = {}
dataset = 'METRLA'
# dataset = 'PEMSBAY'
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join('METRLA\\712', category + '.npz'))
    # cat_data = np.load(os.path.join('BAY', category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']

scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

data['train_loader'] = DataLoader(data['x_train'], data['y_train'], 64, shuffle=True)
data['val_loader'] = DataLoader(data['x_val'], data['y_val'], 64, shuffle=False)
data['test_loader'] = DataLoader(data['x_test'], data['y_test'], 64, shuffle=False)


def evaluate(model, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = model.eval()
        data_iter = data[f'{mode}_loader'].get_iterator()
        losses, ys_true, ys_pred = [], [], []
        outputs,ys = [], []
        for x, y in data_iter:
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y_conv = y[..., 1:]
            y = y[..., 0:1]
            x = x.permute(0, 2, 1, 3)
            output, query = model(x)
            outputs.append(output)
            ys.append(y)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = masked_mae_loss(output, y, scaler, opt)  # masked_mae_loss(y_pred, y_true)
            loss = loss1
            losses.append(loss.item())
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        y_size = data[f'y_{mode}'].shape[0]
        ys, outputs = torch.cat(ys, dim=0)[:y_size], torch.cat(outputs, dim=0)[:y_size]
        ys_true, ys_pred = torch.cat(ys_true, dim=0)[:y_size], torch.cat(ys_pred, dim=0)[:y_size]

        if mode == 'test' or mode == 'val':
            ys_true, ys_pred = ys.permute(1, 0, 2, 3), outputs.permute(1, 0, 2, 3)
            mae = masked_mae_loss(ys_pred, ys_true, scaler, opt).item()
            mape = masked_mape_loss(ys_pred, ys_true, scaler, opt).item()
            rmse = masked_rmse_loss(ys_pred, ys_true, scaler, opt).item()
            mae_3 = masked_mae_loss(ys_pred[2:3], ys_true[2:3], scaler, opt).item()
            mape_3 = masked_mape_loss(ys_pred[2:3], ys_true[2:3], scaler, opt).item()
            rmse_3 = masked_rmse_loss(ys_pred[2:3], ys_true[2:3], scaler, opt).item()
            mae_6 = masked_mae_loss(ys_pred[5:6], ys_true[5:6], scaler, opt).item()
            mape_6 = masked_mape_loss(ys_pred[5:6], ys_true[5:6], scaler, opt).item()
            rmse_6 = masked_rmse_loss(ys_pred[5:6], ys_true[5:6], scaler, opt).item()
            mae_12 = masked_mae_loss(ys_pred[11:12], ys_true[11:12], scaler, opt).item()
            mape_12 = masked_mape_loss(ys_pred[11:12], ys_true[11:12], scaler, opt).item()
            rmse_12 = masked_rmse_loss(ys_pred[11:12], ys_true[11:12], scaler, opt).item()
            if mode == 'test':
                logger.info('testing:')
            else:
                logger.info('validation:')
            if mode == 'test':
                logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae, mape, rmse))
                logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_3, mape_3, rmse_3))
                logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_6, mape_6, rmse_6))
                logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_12, mape_12, rmse_12))
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)

        return mean_loss, ys_true, ys_pred


def traintest_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs =200
    model = get_model()
    print_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader'].get_iterator()
        losses = []
        for x, y in data_iter:
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y_conv = y[..., 1:]
            y = y[..., 0:1]
            optimizer.zero_grad()
            x = x.permute(0, 2, 1, 3)
            output, query = model(x, y, batches_seen)
            loss1 = masked_mae_loss(output, y, scaler, opt)  # masked_mae_loss(y_pred, y_true)
            loss = loss1
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           5)  # gradient clipping - this does it in place
            optimizer.step()
        train_loss = np.mean(losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1,
                   opt.epoch, batches_seen, train_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.6f, %s, %.6f\n" % (
            "epoch", epoch_num+1, "time used", end_time2 - start_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))

        test_loss, _, _ = evaluate(model, 'test')

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == 35:
                print('Early stopping at epoch: %d' % epoch_num)
                break
    print('=' * 35 + 'Best model performance' + '=' * 35)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, _, _ = evaluate(model, 'test')


def main():
    logger.info(opt.dataset, 'training and testing started', time.ctime())
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(opt.dataset, 'training and testing ended', time.ctime())


if __name__ == '__main__':
    main()





