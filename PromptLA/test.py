import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from itertools import cycle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import progress_bar
from D_model.resnet import ResNet18
from D_model.cnn import *
from PSD import *
import random
from torchvision.models import inception_v3
import warnings
warnings.filterwarnings("ignore")

img_shape = (3, 128, 128)
device = torch.device("cuda:3") if torch.cuda.is_available() else 'cpu'

def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # 去掉最后的分类层
    model.eval()
    return model

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(88, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

cwd = os.getcwd()

img_path = os.path.join(cwd, 'data', 'trigger', 'prompt22')
save_dir = os.path.join(cwd, 'checkpoint', 'v1.5', 'prompt22')

transform_test = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ])

# print("loading dataset......")
true_set = datasets.ImageFolder(os.path.join(img_path, 'test_true'), transform=transform_test)
false_set = datasets.ImageFolder(os.path.join(img_path, 'test_false'), transform=transform_test)

true_set = torch.utils.data.Subset(true_set, random.sample(range(len(true_set)), 50))
false_set = torch.utils.data.Subset(false_set, random.sample(range(len(false_set)), 50))

true_loader = torch.utils.data.DataLoader(true_set, batch_size=25, num_workers=0, shuffle=False, drop_last=True)
false_loader = torch.utils.data.DataLoader(false_set, batch_size=25, num_workers=0, shuffle=False, drop_last=True)

# print("loading model......")
discriminator = Discriminator()
# discriminator = ResNet18(num_classes=2)
# discriminator = cnn(num_classes=2)
discriminator.load_state_dict(torch.load(os.path.join(save_dir, 'Discriminator_22_mlp_all_f.ckpt'), map_location=device))
discriminator.to(device)
extractor = get_inception_model().to(device)

# print("start testing......")
discriminator.eval()
true_value = 0
true_correct = 0
false_value = 0
false_correct = 0
true_total = 0
false_total = 0
with torch.no_grad():
    for batch_idx, (true_inputs, _) in enumerate(true_loader):
        
        true_inputs = true_inputs.to(device)

        # is_belong = Variable(torch.FloatTensor(true_inputs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        is_belong = torch.ones(true_inputs.size(0), dtype=torch.long).to(device)

        # true_outputs = discriminator(extractor(true_inputs))
        true_outputs = discriminator(extract_psd_features(true_inputs).to(device))
        # true_result = torch.where(true_outputs > 0.5, 1, 0)
        # true_total += true_outputs.size(0)
        # true_value += true_outputs.cpu().sum().item()
        # true_correct += true_result.eq(is_belong).cpu().sum().item()

        _, predicted = torch.max(true_outputs.data, 1)
        true_total += is_belong.size(0)
        true_correct += predicted.eq(is_belong.data).cpu().sum().item()


    true_avg_value = true_value / true_total
    true_acc = 100. * true_correct / true_total

    for batch_idx, (false_inputs, _) in enumerate(false_loader):
        
        false_inputs = false_inputs.to(device)

        # is_belong = Variable(torch.FloatTensor(false_inputs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        is_belong = torch.ones(false_inputs.size(0), dtype=torch.long).to(device)

        # false_outputs = discriminator(extractor(false_inputs))
        false_outputs = discriminator(extract_psd_features(false_inputs).to(device))
        # false_result = torch.where(false_outputs > 0.5, 1, 0)
        # false_total += false_outputs.size(0)
        # false_value += false_outputs.cpu().sum().item()
        # false_correct += false_result.eq(is_belong).cpu().sum().item()

        _, predicted = torch.max(false_outputs.data, 1)
        false_total += is_belong.size(0)
        false_correct += predicted.eq(is_belong.data).cpu().sum().item()

    false_avg_value = false_value / false_total
    false_acc = 100. * false_correct / false_total
    # print("True_avg_value: %.4f | False_avg_value: %.4f" % (true_avg_value, false_avg_value))
    # print("True_acc: %.3f%%(%d/%d) | False_acc: %.3f%%(%d/%d)" 
    #     % (true_acc, true_correct, true_total, false_acc, false_correct, false_total))
    print("%.3f %.3f" %(true_acc, false_acc))