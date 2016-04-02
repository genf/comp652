from lxml import html
import requests, os, shutil
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

## clear the data folder
if os.path.exists("./rawdata"):
    shutil.rmtree("./rawdata")
os.mkdir("./rawdata")
os.mkdir("./rawdata/repub2016")
os.mkdir("./rawdata/dem2016")

def save_to_file(l, file_prefix):
    for date,url in l:
        d  = file_prefix + "_".join(date.strip().split(" ")).lower()
        d = d.replace(",","")
        if os.path.exists(d):
            d += "_undercard"
        with open(d,'w') as f:
            r = requests.get(url)
            tree = html.fromstring(r.content)
            text = tree.xpath('//span[@class="displaytext"]')
            if not len(text) == 1:
                print("ERROR WITH " + d)
                continue
            f.write(text[0].text_content())


save_to_file(reps, "./rawdata/repub2016/")
save_to_file(dems, "./rawdata/dem2016/")
