import torch
import torch.nn.functional as F
from torchvision import transforms
from project.age_predictor.vgg import pretrained_vgg
from project.config import *

class AgePredictor():
    def __init__(self, model_file):
        self.predictor = pretrained_vgg(model_file, DEVICE).to(DEVICE)

    def preprocess(self, img):
        """prepocess the numpy array image to fit in vgg

        Args:
            img:  numpy array image to be preprocessed

        Returns:
            out: the tensor preprocessed
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392],
                                 std=[1, 1, 1])
        ])
        img = transform(img).unsqueeze(0)
        r, g, b = torch.split(img, 1, 1)
        out = torch.cat((b, g, r), dim=1)
        out = F.interpolate(out,
                            size=(PREDICT_SIZE, PREDICT_SIZE),
                            mode='bilinear')
        out = out * 255.
        return out

    def predict_age(self, img):
        """

        Args:
            img: the PIL Image to be predicted

        Returns:
            pred_age(int): the predicted age of the image
        """
        preprocessed = self.preprocess(img).to(DEVICE)
        with torch.no_grad():
            pred_age = self.predictor(preprocessed)["fc8"]
        # print(pred_age)
        pred_age = torch.argmax(pred_age.view(-1))
        return pred_age
