import mechanicalsoup
import pandas as pd

browser = mechanicalsoup.StatefulBrowser()
start_url = 'https://www.hearthpwn.com/cards'

browser.open(start_url)

page = browser.get_current_page()

columns = {'name':[], 'img_url':[], 'golden_img_url':[], 'type':[], 'have_normal':[], 'have_golden':[]}
url_dataframe = pd.DataFrame(columns=columns)

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
        name = row.find('h3')
        if not name:
            # print('not found')
            continue
        # print(name.string)
        type = row.find_all('li')[0].find('a').string
        print(name.string, type)
        image = row.find('img')
        url = image['src']
        golden_url = image['data-gifurl']

        
        url_dataframe = url_dataframe.append({'name': name.string, 'img_url': url, 'golden_img_url': golden_url, 'type': type, 'have_normal':False, 'have_golden':False}, ignore_index=True)

    next = page.find('a', rel='next')
    if next:
        browser.follow_link(next)

# print(url_dataframe.head())
url_dataframe.to_csv('card_links.csv')
