import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import time
import os


def urlList():
    url = "https://www.indeed.com/resumes?q=movie&co=US&cb=jt&start=400"
    urlList = []
    print('Grabbing the page...')
    response = requests.get(url)
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    print('Collecting the links...')
    for link in links:
        if link.get('href').startswith("/r/") :
            urlList.append( "https://www.indeed.com" +link.get('href') + '\n')

    return urlList

def extract_summary_from_result(soup):
  summaries = []
  spans = soup.findAll('div', attrs={'class': 'hresume'})
  #print(str(spans)[1:-1])

  for span in spans:
    summaries.append(span.text.strip())
  return(summaries)

os.chdir(r"C:\Users\Samuel\Desktop")
os.system("chcp 65001")
i = 144
urlList = urlList()
for url in urlList:
    #print(url)
    URL = url

    page = requests.get(URL)
    soup = BeautifulSoup(page.text, "html.parser")
    file = open(r'C:\Users\Samuel\Desktop\pqpResumes\resume' + str(i) + '.txt', 'w+')
    text = str(extract_summary_from_result(soup))
    text = text.replace(u'\xa0', u' ')
    file.write(text)
    file.close()
    i = i + 1
