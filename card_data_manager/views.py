from django.shortcuts import render, redirect
from django.http import HttpResponse
from card_data_manager.utils.card_repo import HSCardRepo
from card_data_manager.utils.collector import Collector

# Create your views here.
def index(request):
    repo = HSCardRepo()
    cards = repo.GetAll()
    context = {'cards': cards}
    return render(request, 'control_center/index.html', context=context)

def collect(request):
    collector = Collector()
    collector.collect_cards_from_hearthpwn()
    return redirect('/')

def download_collected_cards(request):
    collector = Collector()
    collector.download_collected_normal_cards()
    return redirect('/')