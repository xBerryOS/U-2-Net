import os

from skimage import io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


import numpy as np
from PIL import Image

from tqdm import auto

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert("RGB")
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + ".png")


def main():
    # --------- 1. get image path and name ---------
    model_name = "u2net"  # u2netp

    image_dir = os.path.join(os.getcwd(), "test_data", "test_images")
    prediction_dir = os.path.join(
        os.getcwd(), "test_data", model_name + "_results" + os.sep
    )
    model_dir = os.path.join(
        os.getcwd(), "saved_models", model_name, model_name + ".pth"
    )
    main_path = "/home/piotr/Downloads/carcutter_dataset/background_remove/val/"
    # img_name_list = glob.glob(image_dir + os.sep + "*")
    img_name_list = [main_path + file for file in os.listdir(main_path)]
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab()]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # --------- 3. model define ---------
    if model_name == "u2net":
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load("saved_models/trained_models/u2net_2650_0.06782126222969964.pth")["model"])
        net.cuda()
    else:
        net.load_state_dict(
            torch.load("saved_models/trained_models/u2net_interior2.pth", map_location="cpu")[
                "model"])
    net.eval()
    os.makedirs("saved_images", exist_ok=True)
    # --------- 4. inference for each image ---------
    for i_test, data_test in auto.tqdm(enumerate(test_salobj_dataloader)):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :].detach()
        img = Image.open(img_name_list[i_test])
        pred = F.interpolate(pred.unsqueeze(0), img.size[::-1])[0].permute(1, 2, 0).numpy()
        img = np.asarray(img)
        pred_img = (img * (1 - pred) + pred * np.array([0, 255, 0])).astype("uint8")
        res_img = np.hstack([pred_img, img])
        # img = Image.fromarray((pred[:, :, 0] * 255).astype("uint8"), "L")
        img = Image.fromarray(res_img)
        img.save(f"saved_images2/{img_name_list[i_test].split('/')[-1]}")

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()
