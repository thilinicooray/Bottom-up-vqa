import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math

from dataset_grouped import Dictionary, VQAFeatureDataset_withmask
import pretrained_tda_caq_model, base_model
from train_caq_pretrained_tda import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='caq_newatt')
    parser.add_argument('--pretrained_tda_model', type=str, default='', help='pretrained tda model')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset_withmask('train', dictionary)
    eval_dset = VQAFeatureDataset_withmask('val', dictionary)
    batch_size = args.batch_size

    constructor = 'build_baseline0_newatt'
    baseline = getattr(base_model, constructor)(train_dset, args.num_hid)

    constructor = 'build_%s' % args.model
    model = getattr(pretrained_tda_caq_model, constructor)(train_dset, args.num_hid, baseline).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    utils.load_net(args.pretrained_tda_model, [model.tda_model], ['module'])

    utils.set_trainable(model.tda_model, False)

    model = nn.DataParallel(model).cuda()

    #seventyfive = list(range(0, int(math.ceil(len(train_dset) * 0.75))))
    #trainset_1 = torch.utils.data.Subset(train_dset, seventyfive)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, args.output)
