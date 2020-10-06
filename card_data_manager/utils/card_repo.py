from card_data_manager.models import CardModel
from django.core.files import File
import os

class HSCardRepo:
    def __init__(self):
        return

    def GetAll(self):
        cards = CardModel.objects.all()
        return cards

    def Create(self, **kwargs):
        card = CardModel(**kwargs)
        card.save()
        return card

    def Search(self, **kwargs):
        return CardModel.objects.filter(**kwargs)

    def SetNormalImage(self, card, img_path, save_path):
        old = card.normal_img
        if os.path.isfile(old):
            os.remove(old)
        card.normal_image.save(save_path, File(img_path))
        card.save()
        

