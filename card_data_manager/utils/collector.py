from card_data_manager.utils.card_repo import HSCardRepo
import re
import mechanicalsoup
import requests
import shutil

repo = HSCardRepo()

def collect_cards_from_hearthpwn():
    browser = mechanicalsoup.StatefulBrowser()
    start_url = 'https://www.hearthpwn.com/cards'
    browser.open(start_url)
    page = browser.get_current_page()

    next = page.find('a', rel='next')
    i = 0
    while(next):
        print('page: ', i)
        i += 1
        page = browser.get_current_page()
        table = page.find('table', id='cards').find_all('tr')
        # print(table)
        for row in table:
            # print(row)
            h3 = row.find('h3')
            img = row.find('img')
            # print(name)
            if not h3:
                continue
            # print(name)

            card_id = int(re.findall(r'\d+', h3.find('a')['href'])[0])
            if repo.Search(card_id=card_id).exists():
                continue

            name = h3.string
            img_url = img['data-imageurl']
            gold_img_url = img['data-animationurl']
            card_type = row.find('ul').find('li').find('a').string

            print(name, card_type)

            # card = CardModel(
            #     card_id=card_id,
            #     name=name,
            #     card_type=card_type,
            #     normal_url=img_url,
            #     golden_url=gold_img_url,
            #     normal_dataset=False, 
            #     golden_dataset=False,
            # )
            # card.save()
            repo.Create(
                card_id=card_id,
                name=name,
                card_type=card_type,
                normal_url=img_url,
                golden_url=gold_img_url,
                normal_dataset=False, 
                golden_dataset=False
                )

        next = page.find('a', rel='next')
        if next:
            browser.follow_link(next)


def download_collected_normal_cards():
    cards = CardModel.objects.all()
    for c in cards:
        print(c.normal_url)
    pass