import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
import cv2
from models.moran import MORAN


def load_moran_model(model_path, alphabet, cuda_flag):
    if cuda_flag:
        moran_model = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=True).cuda()
    else:
        moran_model = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=False)

    print('Loading pretrained model from %s' % model_path)
    if cuda_flag:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # Remove `module.`
        MORAN_state_dict_rename[name] = v
    moran_model.load_state_dict(MORAN_state_dict_rename)

    for p in moran_model.parameters():
        p.requires_grad = False
    moran_model.eval()

    return moran_model


def run_demo(moran_model, image_path, alphabet, cuda_flag):
    converter = utils.strLabelConverterForAttention(alphabet, ':')
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(image_path).convert('L')
    image = transformer(image)

    if cuda_flag:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    text = torch.LongTensor(1 * 5)
    length = torch.IntTensor(1)
    text = Variable(text)
    length = Variable(length)

    max_iter = 20
    t, l = converter.encode('0' * max_iter)
    utils.loadData(text, t)
    utils.loadData(length, l)
    output = moran_model(image, length, text, text, test=True, debug=True)

    preds = output[0]  # Extract the prediction tensor from the output tuple
    demo = output[1]
    if isinstance(preds, tuple):
        preds = preds[0]  # If preds is a tuple, take the first element as the prediction tensor

    _, preds = preds.max(1)

    sim_preds = converter.decode(preds.data, length.data)
    sim_preds = sim_preds.strip().split('$')[0]

    return sim_preds, demo


def main():
    model_path = './demo.pth'
    image_path = './ambitiousness.jpg'
    alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'

    cuda_flag = torch.cuda.is_available()

    moran_model = load_moran_model(model_path, alphabet, cuda_flag)
    sim_preds, demo = run_demo(moran_model, image_path, alphabet, cuda_flag)

    print('\nResult:\n' + 'Word is : ' + sim_preds)

    cv2.imshow("demo", demo)
    cv2.waitKey()


if __name__ == "__main__":
    main()
