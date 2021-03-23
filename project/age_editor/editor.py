import torch
import numpy as np
from .generator import pretrained_generator
from torchvision import transforms

class AgeEditor():
    def __init__(self, model_file):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = pretrained_generator(model_file, self.device).to(self.device)
        self.middle_size = 512

    def preprocess(self, img):
        """prepocess the numpy array image to fit in vgg

        Args:
            img: numpy array image to be preprocessed

        Returns:
            out: the tensor preprocessed
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.middle_size, self.middle_size)),
            transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392],
                                 std=[1, 1, 1])
        ])
        out = transform(img).unsqueeze(0)
        return out

    def recover(self, processed):
        bs, c, h, w = processed.size()
        mean = torch.tensor([0.48501961, 0.45795686, 0.40760392]).reshape((3, 1, 1)).repeat(1, h, w)
        img_modif = processed[0].cpu() + mean
        recover_transform = transforms.Compose([
            transforms.ToPILImage()
        ])

        img_modif = recover_transform(img_modif.detach())
        img_modif = np.asarray(img_modif)
        return img_modif


    def edit_age(self, img, dst_age):
        """

        Args:
            img: the numpy array to be predicted

        Returns:
            pred_age(int): the predicted age of the image
        """
        # print(img)

        preprocessed = self.preprocess(img).to(self.device)
        dst_age = torch.tensor([dst_age]).to(self.device)
        with torch.no_grad():
            _, processed, __ = self.generator(preprocessed, None, dst_age)
            # print(preprocessed)
            # print(preprocessed.size())
            img_modif = self.recover(processed)

        return img_modif
