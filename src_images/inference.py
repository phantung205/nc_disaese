from argparse import ArgumentParser
import torch
from src_images import config
from src_images.model import DiabeticRetinopathy
from torchvision.transforms import Compose,ToTensor,Resize
import torch.nn as nn
import cv2
import numpy as np


def get_args():
    parser = ArgumentParser(description="BrainTumorMRI CNN inference")
    parser.add_argument("--image-path","-p",type=str,default=None)
    parser.add_argument("--image-size","-i",type=int,default=config.image_size,help="image size")
    parser.add_argument("--checkpoint","-c",type=str,default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args


def main(args,image_path,image_size):

    # use th GPU if the CPU is unavailable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initialize model
    model = DiabeticRetinopathy().to(device)

    #loader model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("no checkpoint found")
        exit(0)

    # load image
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    #transfrom
    image = cv2.resize(image,(args.image_size,args.image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0

    # add batch size
    image = image[None,:,:,:]
    image = torch.from_numpy(image).to(device).float()

    # turn on model eval
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        print(output)
        probabirity = softmax(output)
        print(probabirity)

    #Extract the vector with the highest score
    max_idx = torch.argmax(probabirity)
    predicted_class = config.categories[max_idx]
    print("the test image is abount {} with confident score of {:.4f}".format(predicted_class, probabirity[0, max_idx]))
    cv2.imshow("{} : {:.2f}%".format(predicted_class, probabirity[0, max_idx] * 100), ori_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()
    main(args, args.image_path, args.image_size)
