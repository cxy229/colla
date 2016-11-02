# 需要author1.sqlite
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
import json
import re
from functools import reduce
from sqlalchemy import create_engine
engine = create_engine('sqlite:///author1.sqlite', echo=False)
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from sqlalchemy import Column, Integer, String , DateTime, ForeignKey
import time
import requests
from bs4 import BeautifulSoup
Base.metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()


urlpt_baseurl = 'http://dblp.uni-trier.de/search/author?xauthor='
author_baseurl = 'http://dblp.uni-trier.de/pers/xx/'
APIKEY = ''
start_index = 3067 # 
end_index = 7490

class Author(Base):
     __tablename__ = 'Author'
     id = Column(Integer, primary_key=True)
     name = Column(String(50), unique=True)
     urlpt = Column(String(50), unique=True)
     begin_time = Column(Integer, default=2222)
     article_num = Column(Integer, default=0)
     article_keys = Column(String(), default=None)
     article_years = Column(String(), default=None)
     coauthor_num = Column(Integer, default=0)
     coauthor_names = Column(String(), default=None)
     coauthor_years = Column(String(), default=None)
     citations_nums = Column(String(), default=None)

     def __init__(self, name, urlpt, begin_time, article_num, article_keys, article_years, coauthor_num, coauthor_names, coauthor_years, citations_nums):
         self.name = name
         self.urlpt = urlpt
         self.begin_time = begin_time

         self.article_num = article_num
         self.article_keys = article_keys
         self.article_years = article_years

         self.coauthor_num = coauthor_num
         self.coauthor_names = coauthor_names
         self.coauthor_years = coauthor_years

         self.citations_nums = citations_nums

     def __repr__(self):
        return "<Author('%d','%s', '%s')>" % (self.id, self.name)

class Colla(Base):
     __tablename__ = 'Colla'
     id = Column(Integer, primary_key=True)
     author_name1 = Column(String(50))
     author_name2 = Column(String(50))

     begin_time = Column(Integer, default=2222)
     colla_time = Column(Integer, default=0)

     coarticle_num = Column(Integer, default=0)
     coarticle_years = Column(String(), default=None)

     scientific_age1 = Column(Integer, default=0)
     scientific_age2 = Column(Integer, default=0)
     article_num1 = Column(Integer, default=0)
     article_num2 = Column(Integer, default=0)
     citation_num1 = Column(Integer, default=-1)
     citation_num2 = Column(Integer, default=-1)
     common_neighbors_num = Column(Integer, default=-1)
     shortest_path_length = Column(Integer, default=2222)


     def __init__(self, author_name1, author_name2, begin_time, colla_time, coarticle_num, coarticle_years,\
                    scientific_age1, scientific_age2, article_num1, article_num2, citation_num1, citation_num2,\
                    common_neighbors_num, shortest_path_length):
         self.author_name1 = author_name1
         self.author_name2 = author_name2

         self.begin_time = begin_time
         self.colla_time = colla_time

        #  self.coarticle_keys = coarticle_keys
         self.coarticle_num = coarticle_num
         self.coarticle_years = coarticle_years

         self.scientific_age1 = scientific_age1
         self.scientific_age2 = scientific_age2
         self.article_num1 = article_num1
         self.article_num2 = article_num2
         self.citation_num1 = citation_num1
         self.citation_num2 = citation_num2
         self.common_neighbors_num = common_neighbors_num
         self.shortest_path_length = shortest_path_length

     def __repr__(self):
        return "<Colla('%s','%s', '%d', '%d')>" % (self.author_name1, self.author_name2, self.begin_time, self.colla_time)


def safe_requests_get(url):
    '''
    解决请求失败问题
    '''
    try:
        r = requests.get(url)
    except Exception as e:
        r = safe_requests_get(url)
    finally:
        # while r.status_code == 429: # 解决 429 Too Many Requests 问题
        #     print('429 Too Many Requests')
        #     time.sleep(2)
        #     r = safe_requests_get(urlpt_baseurl+author_name)
        return r

def driver_execute_script(driver, url, js):
    try:
        driver.execute_script(js)
    except WebDriverException as e:
        print('WebDriverException')
        driver.get(url)
        driver_execute_script(driver,url,js)

def getAuthorId(author_name, article_num):
    list_author_name = author_name.split(' ')
    while '' in list_author_name:
        list_author_name.remove('')
    authfirst = reduce(lambda x,y: x + ' ' + y, list_author_name[:-1])
    authlast = list_author_name[-1]
    url = 'http://api.elsevier.com/content/search/author?query=authfirst('+authfirst+')%20and%20authlast(' + authlast + \
        ')&field=identifier,document-count&apiKey='+ APIKEY +'&httpAccept=application%2Fjson'
    r = safe_requests_get(url)
    data = json.loads(r.text)
    totalResults = int(data['search-results']['opensearch:totalResults'])
    author_id = ''
    if totalResults > 0:    # 搜索有结果
        min_diff = 10000    # 获取author_id
        for entry in data['search-results']['entry']:
            document_count = int(entry['document-count'])
            diff = abs(document_count-article_num)
            if  diff < min_diff:
                min_diff = diff
                author_id = entry['dc:identifier'].strip('AUTHOR_ID:')
    return author_id

def getCitationsNums(author_id):
    driver = webdriver.PhantomJS()
    citations_nums = ''
    if author_id != '':
        url = 'https://www.scopus.com/hirsch/author.uri?accessor=authorProfile&auidList='+author_id+'&origin=AuthorProfile&txGid=0#lvl1Tab2'
        driver.get(url)
        # time.sleep(3)
        js = 'AuthEval.citationTabClick(event);'
        driver_execute_script(driver, url, js)

        time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        list_citDataTable = soup.find_all(id='citDataTable')
        if len(list_citDataTable) == 0:   # 解决没有citDataTable问题
            return citations_nums
        citDataTable = list_citDataTable[0]
        while len(list(citDataTable.children)) == 0:
            print('len(list(citDataTable.children)) == 0')
            driver_execute_script(driver, url, js)
            time.sleep(3)
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            list_citDataTable = soup.find_all(id='citDataTable')
            if len(list_citDataTable) == 0:   # 解决没有citDataTable问题
                return citations_nums
            citDataTable = list_citDataTable[0]

        citations_squences = citDataTable.tbody.find_all('tr')
        # print(citations_squences)
        for sequence in citations_squences:
            tds = sequence.find_all('td')
            # citations_nums += (tds[0].div.text + ':' + tds[1].div.text + '#')
            citations_nums += (tds[1].div.text + '#')
    driver.close()
    return citations_nums

if __name__ == '__main__':
    start_time = time.time()


    for author_index in range(start_index, end_index):
        print('author_index = %r' %author_index)
        author = session.query(Author).filter_by(id = author_index).first()
        author_name = author.name
        article_num = author.article_num
        print(author_name, article_num)
        if author.citations_nums is None:
            author_id = getAuthorId(author_name, article_num)
            print(author_id)
            citations_nums = getCitationsNums(author_id)
            if citations_nums:
                print(citations_nums)
                author.citations_nums = citations_nums
            else:
                print('None')
                author.citations_nums = 'None'
            session.commit()
        end_time = time.time()
        print('time = %r' %(end_time-start_time))
