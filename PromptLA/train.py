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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from utils import *
from D_model.resnet import ResNet18
from D_model.cnn import *
from PSD import *
from torchvision.models import inception_v3
import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd()
img_shape = (3, 128, 128)
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

log_dir = os.path.join(cwd, 'log', 'v1.5', 'train')
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'log_all_f.txt')
set_up_logger(logfile)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(int(np.prod(img_shape)), 512),
            # nn.Linear(2048, 512),
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


def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # 去掉最后的分类层
    model.eval()
    return model


def test(d, true_loader, false_loader, device):
    d.eval()
    true_value = 0
    false_value = 0
    true_total = 0
    false_total = 0
    true_correct = 0
    false_correct = 0
    with torch.no_grad():
        for batch_idx, (true_inputs, _) in enumerate(true_loader):
            
            # is_belong = Variable(torch.FloatTensor(true_inputs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            is_belong = torch.ones(true_inputs.size(0), dtype=torch.long).to(device)

            true_inputs = true_inputs.to(device)
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
            
            # is_belong = Variable(torch.FloatTensor(false_inputs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            is_belong = torch.ones(false_inputs.size(0), dtype=torch.long).to(device)

            false_inputs = false_inputs.to(device)
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
    
    # logging.info("True_avg_value: %.4f | False_avg_value: %.4f" % (true_avg_value, false_avg_value))
    logging.info("True_acc: %.3f%%(%d/%d) | False_acc: %.3f%%(%d/%d)" 
        % (true_acc, true_correct, true_total, false_acc, false_correct, false_total))
    
    return true_avg_value, false_avg_value, true_acc, false_acc

img_path = os.path.join(cwd, 'data', 'trigger', 'prompt22')
save_dir = os.path.join(cwd, 'checkpoint', 'v1.5', 'prompt22')
os.makedirs(save_dir, exist_ok=True)

transform_train = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ])

logging.info("loading dataset......")
train_true_set = datasets.ImageFolder(os.path.join(img_path, 'train_true'), transform=transform_train)
train_false_set = datasets.ImageFolder(os.path.join(img_path, 'train_false_all'), transform=transform_train)
train_true_loader = torch.utils.data.DataLoader(train_true_set, batch_size=20, num_workers=0, shuffle=True, drop_last=True)
train_false_loader = torch.utils.data.DataLoader(train_false_set, batch_size=20, num_workers=0, shuffle=True, drop_last=True)

test_true_set = datasets.ImageFolder(os.path.join(img_path, 'test_true'), transform=transform_train)
test_false_set = datasets.ImageFolder(os.path.join(img_path, 'test_false_all'), transform=transform_train)
test_true_loader = torch.utils.data.DataLoader(test_true_set, batch_size=20, num_workers=0, shuffle=False, drop_last=True)
test_false_loader = torch.utils.data.DataLoader(test_false_set, batch_size=20, num_workers=0, shuffle=False, drop_last=True)

logging.info("loading model......")
# criterion = torch.nn.BCELoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)
discriminator = Discriminator().to(device)
# discriminator = ResNet18(num_classes=2).to(device)
# discriminator = cnn(num_classes=2).to(device)
# extractor = get_inception_model().to(device)

optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
# optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5) # c l
scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

logging.info("start training......")
max_true_acc = 0
min_false_acc = 100
min_loss = 1e5
for epoch in range(60):
    train_losses = []
    train_acc = 0
    total = 0
    correct = 0
    for batch_idx, ((true_inputs, _), (false_inputs, _)) in enumerate(zip(cycle(train_true_loader), train_false_loader)):
        
        true_inputs, false_inputs = true_inputs.to(device), false_inputs.to(device)

        # is_belong = Variable(torch.FloatTensor(true_inputs.size(0),).fill_(1.0), requires_grad=False).to(device)
        # not_belong = Variable(torch.FloatTensor(true_inputs.size(0),).fill_(0.0), requires_grad=False).to(device)

        is_belong = torch.ones(true_inputs.size(0), dtype=torch.long).to(device)
        not_belong = torch.zeros(true_inputs.size(0), dtype=torch.long).to(device)
        
        optimizer.zero_grad()

        # print(true_inputs.size())
        # print(extractor(true_inputs).size())

        # true_outputs = discriminator(extractor(true_inputs))
        # false_outputs = discriminator(extractor(false_inputs))

        true_outputs = discriminator(extract_psd_features(true_inputs).to(device))
        false_outputs = discriminator(extract_psd_features(false_inputs).to(device))

        true_loss = criterion(true_outputs, is_belong)
        false_loss = criterion(false_outputs, not_belong)
        loss = (true_loss + false_loss) / 2
        # loss = 2 * true_loss + false_loss

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # true_result = torch.where(true_outputs > 0.5, 1, 0)
        # false_result = torch.where(false_outputs > 0.5, 1, 0)

        _, true_predicted = torch.max(true_outputs.data, 1)
        _, false_predicted = torch.max(false_outputs.data, 1)
        total += is_belong.size(0)
        correct += true_predicted.eq(is_belong.data).cpu().sum().item()
        total += not_belong.size(0)
        correct += false_predicted.eq(not_belong.data).cpu().sum().item()

    train_acc = 100. * correct / total

    scheduler.step()

    logging.info("Epoch %d/60: loss: %.4f | acc: %.4f%%(%d/%d)" 
            % (epoch, np.average(train_losses), train_acc, correct, total))

    logging.info("Test......")
    true_avg_value, false_avg_value, true_acc, false_acc = test(discriminator, test_true_loader, test_false_loader, device)
    

    if np.average(train_losses) < min_loss:
    # if (true_acc - false_acc) > (max_true_acc - min_false_acc):
        min_loss = np.average(train_losses)
        # max_true_acc = true_acc
        # min_false_acc = false_acc
        logging.info("saving model...")
        torch.save(discriminator.state_dict(), os.path.join(save_dir, 'Discriminator_22_mlp_all_f.ckpt'))