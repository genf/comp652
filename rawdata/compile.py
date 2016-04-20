import numpy as np
import os, itertools, re

pattern = r'([A-Z]+[^a-z:);]+:)'
text = {}
dems = set(['JFK','CLINTON',"O'MALLEY","SANDERS", "CHAFEE", "WEBB"])
reps = set(['REAGAN','BUSH','RUBIO','TRUMP','CRUZ','KASICH','GRAHAM','FIORINA', "CARSON", "CHRISTIE", "PAUL", 'HUCKABEE', 'WALKER', "GRAHAM", "SANTORUM","PERRY", "JINDAL"])
candidates = dems | reps 
for cand in candidates:
	text[cand] = []
def clean(name):
	with open(name,'r') as f:
		g = re.split(pattern, f.read())
		prev = None
		for line in g:
			line = line.strip()
			if line[:-1] in candidates:
				prev = line[:-1]
				continue
			if prev:
				# who needs tabs anyways?
				line.replace("\t"," ")
				if line:
					text[prev].append(line)
				prev = None
		

for f in os.listdir("./dem2016"):
	clean("./dem2016/"+f)

for f in os.listdir("./repub2016"):
	clean("./repub2016/"+ f)


pattern2 = r'(\n|\r){2,}'
text["REAGAN"] = []
for f in os.listdir("./reagan"):
	if not f.endswith(".txt"):
		continue
	with open("./reagan/"+f,mode="r") as f:
		for line in re.split(pattern2, f.read()):
			if line.strip() and not line.startswith("Note:"):
				text['REAGAN'].append(line.strip().replace("\n", " ").replace("\r"," "))

text['JFK']=[]
for f in os.listdir("./jfk"):
	if not f.endswith(".txt"):
		continue
	with open("./jfk/"+f,"r") as f:
		for line in re.split(pattern2, f.read()):
			if line.strip() and not line.startswith("Note:"):
				text['JFK'].append(line.strip().replace("\n"," ").replace("\r"," "))

for k,v in text.items():
        print(k + ":   " + str(len(v)))

cnt = 0 
with open('output.csv','w') as out:
	for cand in text:
		dem_flag = int(cand in dems)
		for line in text[cand]:
			if len(line)<65:
				continue
			## Let's split long documents into sentences
			if len(line.split())>100:
				sentences = [sent.strip() for sent in line.split(".") if len(sent.strip())>5]
				for sent in sentences:
					cnt += 1 
					out.write("\t".join([sent,str(dem_flag),cand]))
					out.write("\n\r")
			else:
				cnt += 1
				out.write("\t".join([line,str(dem_flag),cand]))
				out.write("\n\r")

print "Total size " + str(cnt)
# TODO 
# Wall Street.I  believe <---- gotta add a space there
# Remove [Applause] 
