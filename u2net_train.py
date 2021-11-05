import datetime

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim as optim
import os

from tqdm.auto import tqdm

from data_loader import RescaleT, HorizontalFlip, RandomColorJitter, GaussianBlur
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

# from model import U2NET
from model.u2net_refactor import U2NET_full
from saved_models import TRAINED_MODELS_PATH

USE_CUDA = True

bce_loss = nn.BCELoss(size_average=True)

continue_train = None


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print(
        "l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"
        % (
            loss0.data.item(),
            loss1.data.item(),
            loss2.data.item(),
            loss3.data.item(),
            loss4.data.item(),
            loss5.data.item(),
            loss6.data.item(),
        )
    )
    return loss0, loss


writer = SummaryWriter()

model_name = "u2net"

data_dir = "/home/piotr/Projects/PixelguruDownloader/interior/"
tra_image_dir = os.path.join("train" + os.sep)
tra_label_dir = os.path.join("train_masks" + os.sep)

image_ext = ".jpg"
label_ext = ".jpg"

model_dir = os.path.join(os.getcwd(), "saved_models", model_name + os.sep)

epoch_num = 100000
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = [
    os.path.join(data_dir, tra_image_dir, x)
    for x in os.listdir(data_dir + tra_image_dir)
]

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + "_mask" + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(),
        HorizontalFlip(0.5),
        # RandomColorJitter(),
        GaussianBlur(0.5),
    ]),
)
salobj_dataloader = DataLoader(
    salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1
)

net = U2NET_full()

net.load_state_dict(torch.load(
    TRAINED_MODELS_PATH / "u2net" / "u2net.pth",
    torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
))

if USE_CUDA and torch.cuda.is_available():
    net.cuda()

optimizer = optim.Adam(
    net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
)

training_start_time = datetime.datetime.now()
os.makedirs(TRAINED_MODELS_PATH / str(training_start_time), exist_ok=True)
save_frequency = 10

for epoch in tqdm(range(0, epoch_num)):
    net.train()
    epoch_loss = []

    for i, data in enumerate(salobj_dataloader):
        inputs, labels = data["image"], data["label"]

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if USE_CUDA and torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs_v, labels_v = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())

    if epoch % save_frequency == 0:
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            TRAINED_MODELS_PATH / str(
                training_start_time) / f"u2net_{epoch}_{np.mean(epoch_loss)}.pth"
        )
    writer.add_scalar("training loss", np.mean(epoch_loss), epoch)
