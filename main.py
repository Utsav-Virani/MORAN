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

configObject = argparse.Namespace()
configObject.train_nips = 'PATH_TO_DATASET'
configObject.train_cvpr = 'PATH_TO_DATASET'
configObject.valroot = 'PATH_TO_DATASET'
configObject.workers = 2
configObject.batchSize = 64
configObject.imageHeight = 64
configObject.imageWidth = 200
configObject.targetHeight = 32
configObject.targetWidth = 100
configObject.nh = 256
configObject.numberOfEpoch = 1000
configObject.learningRate = 1
configObject.cuda = True
configObject.numberOfGpus = 1
configObject.MORAN = ''
configObject.alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
configObject.sep = ':'
configObject.experiment = 'PATH_TO_SAVE_MODELS'
configObject.displayInterval = 100
configObject.n_test_disp = 10
configObject.valInterval = 100
configObject.saveInterval = 4000
configObject.adam = False
configObject.beta1 = 0.5
configObject.adadelta = True
configObject.sgd = False
configObject.BidirDecoder = True

# Checking if multi-GPU training is supported
assert configObject.numberOfGpus == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

# Creating experiment directory if it doesn't exist
if configObject.experiment is None:
    configObject.experiment = 'expr'
os.system('mkdir {0}'.format(configObject.experiment))

# Setting random seed for reproducibility
configObject.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", configObject.manualSeed)
random.seed(configObject.manualSeed)
np.random.seed(configObject.manualSeed)
torch.manual_seed(configObject.manualSeed)

# Using cuDNN benchmark for faster training if available
cudnn.benchmark = True

# Checking CUDA availability
if not torch.cuda.is_available():
    assert not configObject.cuda, 'Your device does not support cuda.'

# Printing warning if CUDA device is available but not used
if torch.cuda.is_available() and not configObject.cuda:
    print("You should try -cuda option because your device is compitable")

# Creating training dataset loader
trainingDataset = dataset.lmdbDataset(root=configObject.train_nips,
                                    transform=dataset.resizeNormalize((configObject.imageWidth, configObject.imageHeight)),
                                    reverse=configObject.BidirDecoder)
assert trainingDataset

trainingLoader = torch.utils.data.DataLoader(
    trainingDataset, batch_size=configObject.batchSize,
    shuffle=False, sampler=dataset.randomSequentialSampler(trainingDataset, configObject.batchSize),
    num_workers=int(configObject.workers))

# Creating validation dataset loader
testingDataset = dataset.lmdbDataset(root=configObject.valroot,
                                   transform=dataset.resizeNormalize((configObject.imageWidth, configObject.imageHeight)), reverse=configObject.BidirDecoder)

# Calculating number of classes in the alphabet
numberOfClass = len(configObject.alphabet.split(configObject.sep))
nc = 1

# Creating label converter
converter = utils.strLabelConverterForAttention(configObject.alphabet, configObject.sep)

# Defining CrossEntropy loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Moving model and data to GPU if CUDA is enabled
if configObject.cuda:
    MORAN = MORAN(nc, numberOfClass, configObject.nh, configObject.targetHeight, configObject.targetWidth, BidirDecoder=configObject.BidirDecoder, CUDA=configObject.cuda)
else:
    MORAN = MORAN(nc, numberOfClass, configObject.nh, configObject.targetHeight, configObject.targetWidth, BidirDecoder=configObject.BidirDecoder,
                  inputDataType='torch.FloatTensor', CUDA=configObject.cuda)

# Loading pretrained model if provided
if configObject.MORAN != '':
    print('loading pretrained model from %s' % configObject.MORAN)
    if configObject.cuda:
        state_dict = torch.load(configObject.MORAN)
    else:
        state_dict = torch.load(configObject.MORAN, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        MORAN_state_dict_rename[name] = v
    MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

# Creating tensors for images, texts, and lengths
image = torch.FloatTensor(configObject.batchSize, nc, configObject.imageHeight, configObject.imageWidth)
text = torch.LongTensor(configObject.batchSize * 5)
text_rev = torch.LongTensor(configObject.batchSize * 5)
length = torch.IntTensor(configObject.batchSize)

# Moving tensors to GPU if CUDA is enabled
if configObject.cuda:
    MORAN.cuda()
    MORAN = torch.nn.DataParallel(MORAN, device_ids=range(configObject.numberOfGpus))
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
if configObject.adam:
    optimizer = optim.Adam(MORAN.parameters(), lr=configObject.learningRate, betas=(configObject.beta1, 0.999))
elif configObject.adadelta:
    optimizer = optim.Adadelta(MORAN.parameters(), lr=configObject.learningRate)
elif configObject.sgd:
    optimizer = optim.SGD(MORAN.parameters(), lr=configObject.learningRate, momentum=0.9)
else:
    optimizer = optim.RMSprop(MORAN.parameters(), lr=configObject.learningRate)


def val(dataset, criterion):
    print('---Starting Validation---')
    dataLoader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=configObject.batchSize, num_workers=int(configObject.workers))
    val_Iterator = iter(dataLoader)
    numberOfCorrect = 0
    totalNum = 0
    avgLoss = utils.averager()

    for data in val_Iterator:
        # Performing forward pass and calculating loss
        if configObject.BidirDecoder:
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
    print("[LOG] correct / total: %d / %d, " % (numberOfCorrect, totalNum))
    accuracy = numberOfCorrect / float(totalNum)
    print('[LOG] Test loss: %f, accuracy: %f' % (avgLoss.val(), accuracy))
    return accuracy


def trainBatch():
    for data in trainingLoader:
        if configObject.BidirDecoder:
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
for epoch in range(configObject.numberOfEpoch):

    trainingIterator = iter(trainingLoader)
    i = 0
    while i < len(trainingLoader):

        # Performing validation at specified intervals
        if i % configObject.valInterval == 0:
            for p in MORAN.parameters():
                p.requires_grad = False
            MORAN.eval()

            # Validating and saving model if accuracy improves
            acc_tmp = val(testingDataset, criterion)
            if acc_tmp > acc:
                acc = acc_tmp
                torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                    configObject.experiment, i, str(acc)[:6]))

        # Saving model weights at specified intervals
        if i % configObject.saveInterval == 0:
            torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                configObject.experiment, epoch, i))

        # Setting model to training mode
        for p in MORAN.parameters():
            p.requires_grad = True
        MORAN.train()

        # Training on batch and updating loss averager
        cost = trainBatch()
        avgLoss.add(cost)

        # Printing training progress
        if i % configObject.displayInterval == 0:
            t1 = time.time()
            print('[LOG] Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                  (epoch, configObject.numberOfEpoch, i, len(trainingLoader), avgLoss.val(), t1 - t0)),
            avgLoss.reset()
            t0 = time.time()

        i += 1
