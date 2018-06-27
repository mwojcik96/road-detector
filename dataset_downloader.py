import os
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

input_set = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html'
output_set = 'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html'

for item, directory in zip([input_set, output_set], ['./inputs', './outputs']):
    req = Request(item)
    html_page = urlopen(req)

    soup = BeautifulSoup(html_page, "html5lib")

    links = []
    for link in soup.findAll('a'):
        links.append(link.get('href'))

    os.mkdir(directory)

    for link in links:
        print(link)
        print(link.split('/')[-1])
        f = open('./inputs/' + link.split('/')[-1], 'wb')
        f.write(urlopen(link).read())
        f.close()