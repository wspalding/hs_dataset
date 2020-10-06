from card_data_manager.utils.card_repo import HSCardRepo
from tempfile import NamedTemporaryFile
import re
import mechanicalsoup
import requests
import shutil


class Collector:
    def __init__(self):
        self.CardRepo = HSCardRepo()

    def collect_cards_from_hearthpwn(self):
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
                if self.CardRepo.Search(card_id=card_id).exists():
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
                self.CardRepo.Create(
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


    def download_collected_normal_cards(self):
        cards = self.CardRepo.GetAll()
        i = 0
        for c in cards:
            if i > 2:
                break
            url = c.normal_url
            print(url)
            temp_img = NamedTemporaryFile(delete=True)
            temp_img.write(requests.get(url).content)
            temp_img.flush()
            self.CardRepo.SetNormalImage(c, "{}_normal.png", temp_img)
            i += 1

