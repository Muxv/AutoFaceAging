import torch

VGG_PATH = "../model/dex_imdb_wiki.caffemodel.pt"
# EDITOR_PATH = "../model/checkpoint"
EDITOR_PATH = "../model/hrfaGAN_epoch_10.pth"
UI_PATH = "../ui/aging.ui"


DEVICE = "cuda" if  torch.cuda.is_available() else "cpu"
PREDICT_SIZE = 224
MIDDLE_SIZE = 256
TARGET_SIZE = 256

SERVER_IP = "localhost"


API = f"http://{SERVER_IP}:5555/"
PREDICT_ROUTE = "predict_age"
EDITOR_ROUTE = "edit"