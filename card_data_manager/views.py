from django.shortcuts import render, redirect
from django.http import HttpResponse
from card_data_manager.utils.collector import collect_cards_from_hearthpwn

# Create your views here.
def index(request):
    context = {}
    return render(request, 'index.html', context=context)

def collect(request):
    collect_cards_from_hearthpwn()
    return redirect('/')