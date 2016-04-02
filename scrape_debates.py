from lxml import html
import requests, os 
import pdb
url = "http://www.presidency.ucsb.edu/debates.php"

t = html.fromstring(requests.get(url).content)
trs = t.xpath("//table/tr/td/table/tr")
dems = []
reps = []
for tr in trs:
    tds = tr.getchildren()
    if len(tds) != 2:
        continue
    if not (tds[0].tag == 'td' and tds[1].tag == 'td'):
        continue
    date = tds[0].text_content().lower()
    if not ('2015' in date or '2016' in date):
        continue
#    pdb.set_trace()
    link = tds[1].getchildren()
    if not link:
        continue
    href = link[0].attrib.get('href')
    if not href:
        continue
    if 'republican' in tr.text_content().lower():
        reps.append((date, href))
    elif 'democrat' in tr.text_content().lower():
        dems.append((date,href))

if not os.path.exists("./data"):
    os.mkdir("./data")
if not os.path.exists("./data/repub2016"):
    os.mkdir("./data/repub2016")
if not os.path.exists("./data/dem2016"):
    os.mkdir("./data/dem2016")

for date,url in reps:
    d  ="./data/repub2016/" + "_".join(date.strip().split(" ")).lower()
    with open(d,'w') as f:
        r = requests.get(url)
        tree = html.fromstring(r.content)
        text = tree.xpath('//span[@class="displaytext"]')
        if not len(text) == 1:
            print("ERROR WITH " + d)
            continue
        f.write(text[0].text_content())
for date,url in dems:
    d  ="./data/dem2016/" + "_".join(date.strip().split(" ")).lower()
    with open(d,'w') as f:
        r = requests.get(url)
        tree = html.fromstring(r.content)
        text = tree.xpath('//span[@class="displaytext"]')
        if not len(text) == 1:
            print("ERROR WITH " + d)
            continue
        f.write(text[0].text_content())
