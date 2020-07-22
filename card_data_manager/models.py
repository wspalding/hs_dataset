from django.db import models

card_type_choices = [
    ('m', 'Minion'),
    ('hp', 'Hero Power'),
    ('h', 'Hero'),
    ('ph', 'Playable Hero'),
    ('w', 'Weapon'),
    ('a', 'Ability')
]

# Create your models here.
class CardModel(models.Model):
    card_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    card_type = models.CharField(max_length=20, choices=card_type_choices)
    normal_img = models.ImageField(null=True, blank=True, upload_to='cards/normal/')
    # cropped_normal_img = models.ImageField(null=True, blank=True, upload_to='cards/normal/')
    golden_img = models.ImageField(null=True, blank=True, upload_to='cards/golden/')
    # cropped_golden_img = models.ImageField(null=True, blank=True, upload_to='cards/golden/')
    normal_url = models.CharField(max_length=255, null=True, blank=True)
    golden_url = models.CharField(max_length=255, null=True, blank=True)
    normal_dataset = models.BooleanField()
    golden_dataset = models.BooleanField()
