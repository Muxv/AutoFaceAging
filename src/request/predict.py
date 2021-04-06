import requests
from src.config import API, PREDICT_ROUTE
from io import BytesIO


def predict_request(image_PIL):
    # img = Image.open(file_path)
    byte_io = BytesIO()
    image_PIL.save(byte_io, format="png")
    byte_io.seek(0)

    r = requests.post(API + PREDICT_ROUTE,
                      files={
                          "image": byte_io,
                      })
    return_dict = r.json()
    age = return_dict["age"]
    status = return_dict["status"]
    if status == 200:
        return age
    else:
        return -1