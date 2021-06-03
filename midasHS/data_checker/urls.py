from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # viewing and labeling cards
    path('<int:card_index>', views.card, name='card'),
    # downloading cards
    path('collect', views.collect_new_cards, name='collect'),
    path('download_normal_cards', views.download_normal_cards, name='download_normal'),
    path('download_golden_cards', views.download_golden_cards, name='download_golden'),
    # labeling cards
    path('label/normal', views.label_normal_cards, name='label_normal_cards'),
    path('label/golden', views.label_golden_cards, name='label_golden_cards'),
    # deleting
    path('delete_unneeded', views.delete_unneeded_data, name='delete_unneeded'),
    path('delete_unneeded/cropped', views.delete_cropped_frames, name='delete_cropped_frames'),
    # cropping images
    path('crop_images', views.crop_images, name='crop'),
    path('crop_images/clear_cropped', views.clear_cropped, name='clear_cropped'),
    path('crop_images/to_gif', views.convert_to_gif, name='to_gif'),
    path('crop_images/move', views.move_files, name='move'),
    # misc
    path('misc/rename_data_dirs', views.add_classes_to_card_dirs, name='rename'),
    path('create_dataset', views.convert_to_dataset, name='dataset'),
    path('get_sizes', views.get_data_sizes, name='sizes'),
    path('save_to_file', views.save_to_file, name='save'),
]
