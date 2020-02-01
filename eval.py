import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

from dataset_grouped import Dictionary, VQAFeatureDataset_withmask
import caq_model
from train_caq import evaluate
import utils

def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default='', help='The model we evaluate from')
    parser.add_argument('--model', type=str, default='caq_newatt')
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

    constructor = 'build_%s' % args.model
    model = getattr(caq_model, constructor)(train_dset, args.num_hid).cuda()

    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)


    utils.load_net(args.pretrained_model, [model])

    eval_score, bound, pred_all, qIds = evaluate(model, eval_loader)

    print('eval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

    results = make_json(pred_all, qIds, eval_loader)

    with open(args.output+'/%s_%s.json' \
            % (eval, args.model), 'w') as f:
        json.dump(results, f)