from card_data_manager.models import CardModel

class HSCardRepo:
    def __init__(self):
        return

    def GetAll(self):
        cards = CardModel.objects.all()
        return 

    def Create(self, **kwargs):
        card = CardModel(**kwargs)
        card.save()
        return card

    def Search(self, **kwargs):
        return CardModel.objects.filter(**kwargs)

    def SetImage(self, card, img_path):
        pass

