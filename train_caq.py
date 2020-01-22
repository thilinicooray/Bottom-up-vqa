import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def compute_score_with_logits_paddingremoved(logits, labels):

    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)

    max_labels = torch.max(labels, 1)[1]

    non_padding_idx = (max_labels != (labels.size(1)-1)).nonzero()

    non_padded = torch.index_select(scores.sum(1), 0, non_padding_idx.squeeze())

    final_score = non_padded.sum()

    return final_score, non_padded.size(0)


def train(model, train_loader, eval_loader, num_epochs, output):
    print('training started !')
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    total_steps = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        train_count = 0
        t = time.time()

        for i, (v, b, q, a, m) in enumerate(train_loader):
            total_steps += 1
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            m = Variable(m).cuda()

            v = v.contiguous().view(-1, v.size(2), v.size(3))
            b = b.contiguous().view(-1, b.size(2), b.size(3))
            q = q.contiguous().view(-1, q.size(2))
            a = a.contiguous().view(-1, a.size(2))

            pred = model(v, b, q, a, m)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score, batch_count = compute_score_with_logits_paddingremoved(pred, a.cuda())
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

            train_count += batch_count

            if total_steps % 500 == 0:
                logger.write('train_loss: %.2f, steps:%.2f ' % (total_loss, total_steps))


        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / train_count
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'caq_model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


'''def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    print('evaluating....')

    with torch.no_grad():
        for v, b, q, a, m in iter(dataloader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            m = Variable(m).cuda()

            v = v.contiguous().view(-1, v.size(2), v.size(3))
            b = b.contiguous().view(-1, b.size(2), b.size(3))
            q = q.contiguous().view(-1, q.size(2))
            a = a.contiguous().view(-1, a.size(2))

            pred = model(v, b, q, None, m)
            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound'''


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    count = 0

    print('evaluating....')

    with torch.no_grad():
        for v, b, q, a, m in iter(dataloader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            m = Variable(m).cuda()

            v = v.contiguous().view(-1, v.size(2), v.size(3))
            b = b.contiguous().view(-1, b.size(2), b.size(3))
            q = q.contiguous().view(-1, q.size(2))
            a = a.contiguous().view(-1, a.size(2))

            pred = model(v, b, q, None, m)
            batch_score, batch_count = compute_score_with_logits_paddingremoved(pred, a.cuda())

            score += batch_score
            count += batch_count

            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

    score = score / count

    upper_bound = upper_bound / count
    return score, upper_bound