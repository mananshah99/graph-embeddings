import os
import sys
from tqdm import tqdm
import numpy as np
from time import time

sys.path.insert(0, 'common')
import settings
import util

from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json
import cStringIO
import PIL.Image

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')

def get_image(query):
    query= query.split()
    query='+'.join(query)
    url="https://www.google.com/search?q="+query+"&source=lnms&tbm=isch"
    print url

    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    soup = get_soup(url, header)

    ActualImages=[]# contains the link for Large original images, type of  image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link, Type = json.loads(a.text)["ou"],json.loads(a.text)["ity"]
        ActualImages.append((link,Type))

    print "there are total" , len(ActualImages),"images"

    for i, (img, Type) in enumerate(ActualImages):
        try:
            req = urllib2.Request(img, headers={'User-Agent' : header})

            if len(Type) != 0:
                continue

            raw_img = urllib2.urlopen(req).read()
            raw_csio = cStringIO.StringIO(raw_img)
            img = PIL.Image.open(raw_csio)
            return img

        except Exception as e:
            print "could not load : "+img
            print e
            continue

ids = []

# Fill the ids array
with open(settings.SPECIES_MAPPING, 'r') as f:
    for line in f:
        ids.append(line.rstrip().split('\t'))

def id_to_mapping(id_str):
    for idx in ids:
        # find the id
        if id_str == idx[0]:
            return idx[2]

for idd in ids:
    print idd
#image = get_image('proteobacteria image')
#image.show()

