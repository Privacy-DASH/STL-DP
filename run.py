import os, random, json, time, argparse
import numpy as np
import pandas as pd
from datetime import datetime

from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError

from model import call_by_name

'''
< Utility functions >
'''
def fix_seed(random_seed=2022):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def split_series(series, n_past, n_future):
    '''
    :param series: input time series
    :param n_past: number of past observations
    :param n_future: number of future series
    :return: x, y(label)
    '''
    x, y = list(), list()
    
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future

        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series.iloc[window_start:past_end, :], series.iloc[past_end:future_end, :]
        x.append(past)
        y.append(future)

    x = np.array(x)
    y = np.array(y)
    return x, y

class BaseDataset(nn.Module):
    def __init__(self, inputs, targets):
        super(BaseDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs  = torch.tensor(self.inputs[idx]).type(torch.FloatTensor)
        targets = torch.tensor(self.targets[idx]).type(torch.FloatTensor)

        return inputs, targets

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
< Parsing arguments >
'''
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='CIKM-PAS experiment')
    parser.add_argument('--train',  type=str, default='./data/train.csv')
    parser.add_argument('--valid',  type=str, default='./data/valid.csv')
    parser.add_argument('--test',   type=str, default='./data/test.csv')

    parser.add_argument('--save_root',  type=str, default='./results')
    parser.add_argument('--seed',       type=int, default=2022)
    parser.add_argument('--model',      type=str, default='Linear') # Linear, SimpleDNN, RNNSeq2Seq, TransSeq2Seq

    parser.add_argument('--parallel',           action='store_true')     # Default: False
    parser.add_argument('--scheduler',          action='store_true')     # Default: False
    parser.add_argument('--lr_decay_schedule',  action='store_true')     # Default: False

    parser.add_argument('--past_window',    type=int,   default=60)
    parser.add_argument('--future_window',  type=int,   default=20)
    parser.add_argument('--epoch',          type=int,   default=100)
    parser.add_argument('--batch',          type=int,   default=256)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--weight_decay',   type=float, default=0.0)
    parser.add_argument('--alpha',          type=int,   default=0.5)

    parser.add_argument('--data_type', type=str, default='Original') # Original, LPA_eps_{}, FPA_eps_{}, tFPA_eps_{}, sFPA_eps_{}

    args = parser.parse_args()
    fix_seed(args.seed)

    now = datetime.now()
    time_log = now.strftime('%m%d_%H%M%S')
    save_path = f'{args.save_root}/{time_log}_{args.model}_{args.data_type}'

    '''
    < Preprocessed data -> Window split & Dataloader >
    '''
    train_data = pd.read_csv(args.train)
    valid_data = pd.read_csv(args.valid)
    test_data  = pd.read_csv(args.test)
    forecast_start_idx = 5
    num_forecast = 3

    args.in_units = train_data.shape[1]
    args.out_units = num_forecast

    print('Slicing train dataframe')
    train_sliced_x, train_sliced_y = split_series(train_data, args.past_window, args.future_window)
    train_inputs, train_targets = train_sliced_x, train_sliced_y[:, :, forecast_start_idx:forecast_start_idx+num_forecast] # (36608,60,24), (36608,20,3)

    print('Slicing valid dataframe')
    valid_sliced_x, valid_sliced_y = split_series(valid_data, args.past_window, args.future_window)
    valid_inputs, valid_targets = valid_sliced_x, valid_sliced_y[:, :, forecast_start_idx:forecast_start_idx+num_forecast] # (5162,60,24), (5162,20,3)

    print('Slicing test dataframe')
    test_sliced_x, test_sliced_y = split_series(test_data, args.past_window, args.future_window)
    test_inputs, test_targets = test_sliced_x, test_sliced_y[:, :, forecast_start_idx:forecast_start_idx+num_forecast] # (10409,60,24), (10409,20,3)

    trainSet = BaseDataset(train_inputs, train_targets)
    validSet = BaseDataset(valid_inputs, valid_targets)
    testSet = BaseDataset(test_inputs , test_targets)

    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch, shuffle=True, drop_last=False, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(validSet, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=args.num_workers//2)
    test_loader = torch.utils.data.DataLoader(testSet, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=args.num_workers//2)


    '''
    < Loading model >
    '''
    model = call_by_name(args)
    model.cuda()
    print(f'\nLoad Complete: {args.model}\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')

    os.makedirs(save_path, exist_ok=True)

    '''
    < Train & Validate & Test >
    '''
    valid_best = 99999
    train_best = 0
    test_best = 0
    epoch_best = 0

    for epoch in range(1, args.epoch+1):
        
        if args.scheduler:
            adjust_learning_rate(optimizer, epoch, args)
        
        # Training Process
        train_loss = AverageMeter()
        model.train()
        for iter_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epoch}"):
            inputs  = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), inputs.shape[0])

        # Validation Process
        model.eval()
        with torch.no_grad():
            valid_loss = AverageMeter()
            for iter_idx, (inputs, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validation"):
                inputs  = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss.update(loss.item(), inputs.shape[0])

        # Test Process
            test_loss = AverageMeter()
            for iter_idx, (inputs, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test"):
                inputs  = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss.update(loss.item(), inputs.shape[0])

        if valid_loss.avg < valid_best:
            print(f"Lower validation loss, saving the model.. {valid_best} -> {valid_loss.avg}")
            
            valid_best = valid_loss.avg
            train_best = train_loss.avg
            test_best = test_loss.avg
            epoch_best = epoch
            
            torch.save(model, f'{save_path}/best.pt')

        torch.save(model, f'{save_path}/last.pt')

        print(f"Current epoch {epoch} | Train loss {train_loss.avg}, Valid loss {valid_loss.avg}, Test loss {test_loss.avg}")
        print(f"Best epoch {epoch_best} | Train loss {train_best}, Valid loss {valid_best}, Test loss {test_best}\n")

    print("DONE training")
    f = open(f'{save_path}/train_log.txt', 'w')
    f.write(f"Current epoch {epoch} | Train loss {train_loss.avg}, Valid loss {valid_loss.avg}, Test loss {test_loss.avg}\n")
    f.write(f"Best epoch {epoch_best} | Train loss {train_best}, Valid loss {valid_best}, Test loss {test_best}\n")
    f.close()

    '''
    < Evaluation - MAE, MAPE >
    '''
    eval_loader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=False, drop_last=False, num_workers=args.num_workers//2)

    eval_model = torch.load(f'{save_path}/best.pt')
    eval_model.cuda()
    print(f'\nLoad Complete\n')

    MAE_metric = nn.L1Loss()
    MAPE_metric = MeanAbsolutePercentageError().cuda()

    # Test Process
    MAE = AverageMeter()
    MAPE = AverageMeter()
    # output_list = []
    # label_list = []

    for iter_idx, (inputs, labels) in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluation"):
        inputs  = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        outputs = model(inputs)
        MAE_ = MAE_metric(outputs, labels)
        MAPE_ = MAPE_metric(outputs, labels)

        MAE.update(MAE_.item(), inputs.shape[0])
        MAPE.update(MAPE_.item(), inputs.shape[0])

        # if len(output_list) <= 0:
        #     output_list = outputs
        #     label_list = labels
        # else:
        #     output_list = torch.cat((output_list, outputs), dim=0)
        #     label_list = torch.cat((label_list, labels), dim=0)

    # total_MAE = MAE_metric(output_list, label_list)
    # total_MAPE = MAPE_metric(output_list, label_list)

    print("DONE evaluation")
    f = open(f'{save_path}/eval_log.txt', 'w')
    f.write(f"Eval result | MAE {MAE.avg}, MAPE {MAPE.avg}\n")
    # f.write(f"Eval result | total MAE {total_MAE}, total MAPE {total_MAPE}\n")
    f.close()
