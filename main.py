# Importing necessary libraries and modules
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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Importing custom modules and functions
from tools import dataset
from tools import utils
from models.moran import MORAN

# Parsing command-line arguments
argParser = argparse.ArgumentParser()
argParser.add_argument('--train_nips', required=True, default='/content/data/nips_data', help='path to NIPS dataset')
argParser.add_argument('--train_cvpr', required=True, help='path to CVPR dataset')
argParser.add_argument('--valroot', required=True, help='path to validation dataset')
argParser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
argParser.add_argument('--batchSize', type=int, default=64, help='input batch size')
argParser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
argParser.add_argument('--imgW', type=int, default=200, help='the width of the input image to network')
argParser.add_argument('--targetH', type=int, default=32, help='the height of the output image from network')
argParser.add_argument('--targetW', type=int, default=100, help='the width of the output image from network')
argParser.add_argument('--nh', type=int, default=256, help='size of the LSTM hidden state')
argParser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
argParser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic')
argParser.add_argument('--cuda', action='store_true', help='enables CUDA')
argParser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
argParser.add_argument('--MORAN', default='', help="path to model (to continue training)")
argParser.add_argument('--alphabet', type=str, default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
argParser.add_argument('--sep', type=str, default=':')
argParser.add_argument('--experiment', default='/content/output333/', help='where to store samples and models')
argParser.add_argument('--displayInterval', type=int, default=100, help='interval to be displayed')
argParser.add_argument('--n_test_disp', type=int, default=10, help='number of samples to display when testing')
argParser.add_argument('--valInterval', type=int, default=1000, help='interval to perform validation')
argParser.add_argument('--saveInterval', type=int, default=40000, help='interval to save model weights')
argParser.add_argument('--adam', action='store_true', help='whether to use Adam optimizer (default is RMSprop)')
argParser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer (default=0.5)')
argParser.add_argument('--adadelta', action='store_true', help='whether to use Adadelta optimizer (default is RMSprop)')
argParser.add_argument('--sgd', action='store_true', help='whether to use SGD optimizer (default is RMSprop)')
argParser.add_argument('--BidirDecoder', action='store_true', help='whether to use Bidirectional Decoder')

# Parse arguments
optParser = argParser.parse_args()

# Checking if multi-GPU training is supported
assert optParser.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

# Creating experiment directory if it doesn't exist
if optParser.experiment is None:
    optParser.experiment = 'expr'
os.system('mkdir {0}'.format(optParser.experiment))

# Setting random seed for reproducibility
optParser.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", optParser.manualSeed)
random.seed(optParser.manualSeed)
np.random.seed(optParser.manualSeed)
torch.manual_seed(optParser.manualSeed)

# Using cuDNN benchmark for faster training if available
cudnn.benchmark = True

# Checking CUDA availability
if not torch.cuda.is_available():
    assert not optParser.cuda, 'Your device does not support cuda.'

# Printing warning if CUDA device is available but not used
if torch.cuda.is_available() and not optParser.cuda:
    print("You should try -cuda option because your device is compitable")

# Creating training dataset loader
trainingDataset = dataset.lmdbDataset(root=optParser.train_nips,
                                    transform=dataset.resizeNormalize((optParser.imgW, optParser.imgH)),
                                    reverse=optParser.BidirDecoder)
assert trainingDataset

trainingLoader = torch.utils.data.DataLoader(
    trainingDataset, batch_size=optParser.batchSize,
    shuffle=False, sampler=dataset.randomSequentialSampler(trainingDataset, optParser.batchSize),
    num_workers=int(optParser.workers))

# Creating validation dataset loader
testingDataset = dataset.lmdbDataset(root=optParser.valroot,
                                   transform=dataset.resizeNormalize((optParser.imgW, optParser.imgH)), reverse=optParser.BidirDecoder)

# Calculating number of classes in the alphabet
numberOfClass = len(optParser.alphabet.split(optParser.sep))
nc = 1

# Creating label converter
converter = utils.strLabelConverterForAttention(optParser.alphabet, optParser.sep)

# Defining CrossEntropy loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Moving model and data to GPU if CUDA is enabled
if optParser.cuda:
    MORAN = MORAN(nc, numberOfClass, optParser.nh, optParser.targetH, optParser.targetW, BidirDecoder=optParser.BidirDecoder, CUDA=optParser.cuda)
else:
    MORAN = MORAN(nc, numberOfClass, optParser.nh, optParser.targetH, optParser.targetW, BidirDecoder=optParser.BidirDecoder,
                  inputDataType='torch.FloatTensor', CUDA=optParser.cuda)

# Loading pretrained model if provided
if optParser.MORAN != '':
    print('loading pretrained model from %s' % optParser.MORAN)
    if optParser.cuda:
        state_dict = torch.load(optParser.MORAN)
    else:
        state_dict = torch.load(optParser.MORAN, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        MORAN_state_dict_rename[name] = v
    MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

# Creating tensors for images, texts, and lengths
image = torch.FloatTensor(optParser.batchSize, nc, optParser.imgH, optParser.imgW)
text = torch.LongTensor(optParser.batchSize * 5)
text_rev = torch.LongTensor(optParser.batchSize * 5)
length = torch.IntTensor(optParser.batchSize)

# Moving tensors to GPU if CUDA is enabled
if optParser.cuda:
    MORAN.cuda()
    MORAN = torch.nn.DataParallel(MORAN, device_ids=range(optParser.ngpu))
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
text_rev = Variable(text_rev)
length = Variable(length)

# Loss averager for tracking training loss
avgLoss = utils.averager()

# Setting up optimizer
if optParser.adam:
    optimizer = optim.Adam(MORAN.parameters(), lr=optParser.lr, betas=(optParser.beta1, 0.999))
elif optParser.adadelta:
    optimizer = optim.Adadelta(MORAN.parameters(), lr=optParser.lr)
elif optParser.sgd:
    optimizer = optim.SGD(MORAN.parameters(), lr=optParser.lr, momentum=0.9)
else:
    optimizer = optim.RMSprop(MORAN.parameters(), lr=optParser.lr)


def val(dataset, criterion):
    """
    Validation function for evaluating model performance on validation dataset.

    Args:
        dataset (torch.utils.data.Dataset): Validation dataset.
        criterion (torch.nn.Module): Loss criterion.

    Returns:
        float: Accuracy achieved on the validation dataset.
    """
    print('Start val')
    dataLoader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=optParser.batchSize, num_workers=int(optParser.workers))
    val_Iterator = iter(dataLoader)
    numberOfCorrect = 0
    totalNum = 0
    avgLoss = utils.averager()

    for data in val_Iterator:
        # Performing forward pass and calculating loss
        if optParser.BidirDecoder:
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

        # Calculating accuracy and updating loss averager
        avgLoss.add(cost)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                numberOfCorrect += 1
            totalNum += 1

    # Printing validation results
    print("correct / total: %d / %d, " % (numberOfCorrect, totalNum))
    accuracy = numberOfCorrect / float(totalNum)
    print('Test loss: %f, accuracy: %f' % (avgLoss.val(), accuracy))
    return accuracy


def trainBatch():
    """
    Function for training on a batch of data.

    Returns:
        torch.Tensor: Loss for the batch.
    """
    for data in trainingLoader:
        if optParser.BidirDecoder:
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

        # Zeroing gradients, performing backpropagation, and updating weights
        MORAN.zero_grad()
        cost.backward()
        optimizer.step()
        return cost


# Initializing time and accuracy variables
t0 = time.time()
acc = 0
acc_tmp = 0

# Training loop
for epoch in range(optParser.niter):

    trainingIterator = iter(trainingLoader)
    i = 0
    while i < len(trainingLoader):

        # Performing validation at specified intervals
        if i % optParser.valInterval == 0:
            for p in MORAN.parameters():
                p.requires_grad = False
            MORAN.eval()

            # Validating and saving model if accuracy improves
            acc_tmp = val(testingDataset, criterion)
            if acc_tmp > acc:
                acc = acc_tmp
                torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                    optParser.experiment, i, str(acc)[:6]))

        # Saving model weights at specified intervals
        if i % optParser.saveInterval == 0:
            torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                optParser.experiment, epoch, i))

        # Setting model to training mode
        for p in MORAN.parameters():
            p.requires_grad = True
        MORAN.train()

        # Training on batch and updating loss averager
        cost = trainBatch()
        avgLoss.add(cost)

        # Printing training progress
        if i % optParser.displayInterval == 0:
            t1 = time.time()
            print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                  (epoch, optParser.niter, i, len(trainingLoader), avgLoss.val(), t1 - t0)),
            avgLoss.reset()
            t0 = time.time()

        i += 1
