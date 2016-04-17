# -*- coding: windows-1252 -*
from lxml import html
import os 

def clean_and_save(file):
	with open(file, mode="r", encoding="windows-1252") as f:
		try:
			t = html.fromstring(f.read())
		except Exception as e:
			print(file)
			raise e 
		ps = t.xpath("//p")
		title = t.xpath("//title")
		text = [p.text_content() for p in ps]
	title = "_".join(title[0].text_content().split(" "))
	title = title.replace("?","").replace("-","").replace("\n","")	
	with open(title+".txt","w") as out:
		out.write("\n\r".join(text))		
		

for f in os.listdir("."):
	if f.endswith(".htm"):
		clean_and_save(f)
