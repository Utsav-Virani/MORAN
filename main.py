# from __future__ import print_function

import argparse
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from tools import dataset
from tools import utils
from models.moran import MORAN

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# # Define arguments
# args = """
# --train_nips /content/ic03_867_data
# --train_cvpr /content/ic03_867_data
# --valroot /content/ic03_867_data
# --workers 2
# --batchSize 64
# --niter 10
# --lr 1
# --cuda
# --experiment /content/output333/
# --displayInterval 100
# --valInterval 1000
# --saveInterval 40000
# --adadelta
# --BidirDecoder
# """

# Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--train_nips', required=True, default='/content/data/nips_data', help='path to NIPS dataset')
# parser.add_argument('--train_cvpr', required=True, help='path to CVPR dataset')
# parser.add_argument('--valroot', required=True, help='path to validation dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
# parser.add_argument('--imgW', type=int, default=200, help='the width of the input image to network')
# parser.add_argument('--targetH', type=int, default=32, help='the height of the output image from network')
# parser.add_argument('--targetW', type=int, default=100, help='the width of the output image from network')
# parser.add_argument('--nh', type=int, default=256, help='size of the LSTM hidden state')
# parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic')
# parser.add_argument('--cuda', action='store_true', help='enables CUDA')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--MORAN', default='', help="path to model (to continue training)")
# parser.add_argument('--alphabet', type=str, default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
# parser.add_argument('--sep', type=str, default=':')
# parser.add_argument('--experiment', default='/content/output333/', help='where to store samples and models')
# parser.add_argument('--displayInterval', type=int, default=100, help='interval to be displayed')
# parser.add_argument('--n_test_disp', type=int, default=10, help='number of samples to display when testing')
# parser.add_argument('--valInterval', type=int, default=1000, help='interval to perform validation')
# parser.add_argument('--saveInterval', type=int, default=40000, help='interval to save model weights')
# parser.add_argument('--adam', action='store_true', help='whether to use Adam optimizer (default is RMSprop)')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer (default=0.5)')
# parser.add_argument('--adadelta', action='store_true', help='whether to use Adadelta optimizer (default is RMSprop)')
# parser.add_argument('--sgd', action='store_true', help='whether to use SGD optimizer (default is RMSprop)')
# parser.add_argument('--BidirDecoder', action='store_true', help='whether to use Bidirectional Decoder')

# Parse provided args
# opt = parser.parse_args(args.split())


# parser = argparse.ArgumentParser()

# # Define arguments
# args = """
# --train_nips /content/ic03_867_data
# --train_cvpr /content/ic03_867_data
# --valroot /content/ic03_867_data
# --workers 2
# --batchSize 64
# --niter 10
# --lr 1
# --cuda
# --experiment /content/output333/
# --displayInterval 100
# --valInterval 1000
# --saveInterval 40000
# --adadelta
# --BidirDecoder
# """

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_nips', required=True, default='/content/data/nips_data', help='path to NIPS dataset')
parser.add_argument('--train_cvpr', required=True, help='path to CVPR dataset')
parser.add_argument('--valroot', required=True, help='path to validation dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=200, help='the width of the input image to network')
parser.add_argument('--targetH', type=int, default=32, help='the height of the output image from network')
parser.add_argument('--targetW', type=int, default=100, help='the width of the output image from network')
parser.add_argument('--nh', type=int, default=256, help='size of the LSTM hidden state')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic')
parser.add_argument('--cuda', action='store_true', help='enables CUDA')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--MORAN', default='', help="path to model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
parser.add_argument('--sep', type=str, default=':')
parser.add_argument('--experiment', default='/content/output333/', help='where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='number of samples to display when testing')
parser.add_argument('--valInterval', type=int, default=1000, help='interval to perform validation')
parser.add_argument('--saveInterval', type=int, default=40000, help='interval to save model weights')
parser.add_argument('--adam', action='store_true', help='whether to use Adam optimizer (default is RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer (default=0.5)')
parser.add_argument('--adadelta', action='store_true', help='whether to use Adadelta optimizer (default is RMSprop)')
parser.add_argument('--sgd', action='store_true', help='whether to use SGD optimizer (default is RMSprop)')
parser.add_argument('--BidirDecoder', action='store_true', help='whether to use Bidirectional Decoder')

# Parse provided args
opt = parser.parse_args()


assert opt.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if not torch.cuda.is_available():
    assert not opt.cuda, 'You don\'t have a CUDA device.'

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.train_nips,
                                         transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),
                                         reverse=opt.BidirDecoder)
assert train_dataset
# train_cvpr_dataset = dataset.lmdbDataset(root=opt.train_cvpr,
#                                          transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),
#                                          reverse=opt.BidirDecoder)
# assert train_cvpr_dataset

# train_dataset = torch.utils.data.ConcatDataset([train_nips_dataset, train_cvpr_dataset])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.batchSize),
    num_workers=int(opt.workers))

test_dataset = dataset.lmdbDataset(root=opt.valroot,
                                   transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)

nclass = len(opt.alphabet.split(opt.sep))
nc = 1

converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
criterion = torch.nn.CrossEntropyLoss()

if opt.cuda:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, CUDA=opt.cuda)
else:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder,
                  inputDataType='torch.FloatTensor', CUDA=opt.cuda)

if opt.MORAN != '':
    print('loading pretrained model from %s' % opt.MORAN)
    if opt.cuda:
        state_dict = torch.load(opt.MORAN)
    else:
        state_dict = torch.load(opt.MORAN, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        MORAN_state_dict_rename[name] = v
    MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

image = torch.FloatTensor(opt.batchSize, nc, opt.imgH, opt.imgW)
text = torch.LongTensor(opt.batchSize * 5)
text_rev = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    MORAN.cuda()
    MORAN = torch.nn.DataParallel(MORAN, device_ids=range(opt.ngpu))
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
text_rev = Variable(text_rev)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
elif opt.sgd:
    optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
else:
    optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)


def val(dataset, criterion):
    print('Start val')
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))  # opt.batchSize
    val_iter = iter(data_loader)
    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()

    for data in val_iter:
        if opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            preds0, preds1 = MORAN(image, length, text, text_rev, test=True)
            cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
            preds0_prob, preds0 = preds0.max(1)
            preds0 = preds0.view(-1)
            preds0_prob = preds0_prob.view(-1)
            sim_preds0 = converter.decode(preds0.data, length.data)
            preds1_prob, preds1 = preds1.max(1)
            preds1 = preds1.view(-1)
            preds1_prob = preds1_prob.view(-1)
            sim_preds1 = converter.decode(preds1.data, length.data)
            sim_preds = []
            for j in range(cpu_images.size(0)):
                text_begin = 0 if j == 0 else length.data[:j].sum()
                if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).item() > \
        torch.mean(preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).data:


                    sim_preds.append(sim_preds0[j].split('$')[0] + '$')
                else:
                    sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = MORAN(image, length, text, text_rev, test=True)
            cost = criterion(preds, text)
            _, preds = preds.max(1)
            preds = preds.view(-1)
            sim_preds = converter.decode(preds.data, length.data)

        loss_avg.add(cost)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
            n_total += 1

    print("correct / total: %d / %d, " % (n_correct, n_total))
    accuracy = n_correct / float(n_total)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))
    return accuracy


def trainBatch():
    for data in train_loader:
        if opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            preds0, preds1 = MORAN(image, length, text, text_rev)
            cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = MORAN(image, length, text, text_rev)
            cost = criterion(preds, text)

        MORAN.zero_grad()
        cost.backward()
        optimizer.step()
        return cost


t0 = time.time()
acc = 0
acc_tmp = 0
for epoch in range(opt.niter):

    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):

        if i % opt.valInterval == 0:
            for p in MORAN.parameters():
                p.requires_grad = False
            MORAN.eval()

            acc_tmp = val(test_dataset, criterion)
            if acc_tmp > acc:
                acc = acc_tmp
                torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                    opt.experiment, i, str(acc)[:6]))

        if i % opt.saveInterval == 0:
            torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                opt.experiment, epoch, i))

        for p in MORAN.parameters():
            p.requires_grad = True
        MORAN.train()

        cost = trainBatch()
        loss_avg.add(cost)

        if i % opt.displayInterval == 0:
            t1 = time.time()
            print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val(), t1 - t0)),
            loss_avg.reset()
            t0 = time.time()

        i += 1