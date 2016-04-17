from lxml import html
import os 


def clean_and_save(file):
	with open(file, "r") as f:
		t = html.fromstring(f.read())
		ps = t.xpath("//p")
		text = [p.text_content() for p in ps]
	file = file[:-3]
	with open("clean_"+file+"txt","w") as out:
		out.write("\n\r".join(text))		
		

for f in os.listdir("."):
	if f.endswith(".htm"):
		clean_and_save(f)
