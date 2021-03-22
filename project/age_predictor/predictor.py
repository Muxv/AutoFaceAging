import torch
import torch.nn.functional as F
from .vgg import pretrained_vgg
from torchvision import transforms

class AgePredictor():
    def __init__(self, model_file, target_size=224):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = pretrained_vgg(model_file, self.device).to(self.device)
        self.target_size = target_size


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
                            size=(self.target_size, self.target_size),
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
        preprocessed = self.preprocess(img).to(self.device)
        with torch.no_grad():
            pred_age = self.predictor(preprocessed)["fc8"]
        # print(pred_age)
        pred_age = torch.argmax(pred_age.view(-1))
        return pred_age
