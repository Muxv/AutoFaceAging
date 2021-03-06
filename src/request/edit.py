import requests
from PIL import Image
from src.config import API, EDITOR_ROUTE
from io import BytesIO


def edit_request(image_PIL, dst_age):
    byte_io = BytesIO()
    image_PIL.save(byte_io, format="png")
    byte_io.seek(0)

    r = requests.post(API + EDITOR_ROUTE,
                      files={
                          "image": byte_io,
                      },
                      data={
                          "age": dst_age
                      })

    return_img = Image.open(BytesIO(r.content))
    return return_img


