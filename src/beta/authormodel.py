WWW_NUMS = [7742,7748,8877,12365,12782,14875,14968,16666]
SQL_NAME = 'author3333.sqlite'

from sqlalchemy import create_engine, func

engine = create_engine('sqlite:///../../sql/' + SQL_NAME, echo=False)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from bs4 import BeautifulSoup
import requests
import time
import os

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

urlpt_baseurl = 'http://dblp.uni-trier.de/search/author?xauthor='
author_baseurl = 'http://dblp.uni-trier.de/pers/xx/'
article_baseurl = 'http://dblp.uni-trier.de/rec/xml/'
APIKEY = 'b3a71de2bde04544495881ed9d2f9c5b'


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

    def __init__(self, name, urlpt, begin_time, article_num, article_keys, article_years, coauthor_num, coauthor_names,
                 coauthor_years, citations_nums):
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
        return "<Author('%d','%s')>" % (self.id, self.name)


class Colla(Base):
    '''
     sqlalchemy Colla table
     '''
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

    def __init__(self, author_name1, author_name2, begin_time, colla_time, coarticle_num, coarticle_years, \
                 scientific_age1, scientific_age2, article_num1, article_num2, citation_num1, citation_num2, \
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
        return "<Colla('%s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)


class AuthorFormatted(object):
    '''
    class Author format
    '''

    def __init__(self, name, urlpt, begin_time, article_num, article_keys, article_years, coauthor_num, coauthor_names,
                 coauthor_years, citations_nums):
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


def safe_requests_get(url):
    '''
    解决请求失败问题
    '''
    try:
        r = requests.get(url)
    except Exception as e:
        print('requests.get error')
        r = safe_requests_get(url)
    finally:
        while r.status_code == 429:  # 解决 429 Too Many Requests 问题
            print('429 Too Many Requests')
            time.sleep(2)
            r = safe_requests_get(url)
        return r


def split_sharp(s):
    '''
    分割字符串
    '''
    if type(s) == str:
        if s == 'None':
            return 'None'
        list_s = s.split('#')
        while '' in list_s:  # 去除list中空字符
            list_s.remove('')
        return list_s
    else:
        return None


def split_sharp_int(s):
    '''
    分割字符串，并将每个元素转换成int
    '''
    if type(s) == str:
        if s == 'None':
            return 'None'
        list_s = s.split('#')
        while '' in list_s:  # 去除list中空字符
            list_s.remove('')
        return [int(i) for i in list_s]
    else:
        return None


def driver_execute_script(driver, url, js):
    '''
    解决selenium driver执行脚本问题
    '''
    try:
        driver.execute_script(js)
    except WebDriverException as e:
        print('WebDriverException')
        driver.get(url)
        driver_execute_script(driver, url, js)


def author_format(author, session):
    '''
    把author格式化
    Returns:
        AuthorFormatted
        None:
    '''
    if type(author) == Author:
        coauthor_names = split_sharp(author.coauthor_names)
        coauthor_years = split_sharp(author.coauthor_years)
        coauthor_years = [int(year) for year in coauthor_years]  # year 转换成 int

        article_keys = split_sharp(author.article_keys)
        article_years = split_sharp(author.article_years)
        article_years = [int(year) for year in article_years]  # year 转换成 int

        citations_nums = None

        return AuthorFormatted(author.name, author.urlpt, author.begin_time, author.article_num, article_keys,
                               article_years, author.coauthor_num, coauthor_names, coauthor_years, citations_nums)
    else:
        return None


def get_author(author_name, session):
    '''
    获取author信息，并存到数据库中
    没有返回None，找到了返回 author
    '''
    if author_name is None:
        return None
    begin_time = 2222
    article_keys = ''
    article_years = ''
    coauthor_names = ''
    coauthor_years = ''
    author = session.query(Author).filter_by(name=author_name).first()  # 解决重复的author
    if author is None:
        r = safe_requests_get(urlpt_baseurl + author_name)
        soup = BeautifulSoup(r.text, 'lxml')
        if soup.find('author'):  # 解决 根据 author_name 找不到urlpt问题 之 没有author标签
            l = soup.find_all('author')
        else:
            return None
        i = 0
        while i < len(l):  # urlpt 解决 根据 author_name 找不到urlpt问题 之 没有author标签匹配author_name
            if l[i].text == author_name:
                urlpt = l[i].attrs['urlpt']
                break
            i += 1
        if i == len(l):
            return None

        r = safe_requests_get(author_baseurl + urlpt)
        soup = BeautifulSoup(r.text, 'lxml')
        try:
            if soup.dblpperson.has_attr(
                    'n'):  # 解决 名字问题 http://dblp.uni-trier.de/search/author?xauthor=Roberto%20Zangr&#243;niz
                article_num = int(soup.dblpperson.attrs['n'])
            elif soup.dblpperson.has_attr('f'):
                urlpt = soup.dblpperson.attrs['f']
                r = safe_requests_get(author_baseurl + urlpt)
                soup = BeautifulSoup(r.text, 'lxml')
                article_num = int(soup.dblpperson.attrs['n'])

            if article_num < 10:  # 筛选 论文数量 >= 10
                return None
        except Exception as e:
            print('article_num error')
            return None

        list_articles = soup.find_all('r')  # article
        for article in list_articles:
            article = article.contents[0]
            article_key = article.attrs['key']
            article_keys += (article_key + '#')
            article_year = article.year.text
            if begin_time > int(article_year):  # begin_time
                begin_time = int(article_year)
            article_years += (article_year + '#')

        list_coauthors = soup.find_all('co')  # coauthor
        coauthor_num = len(list_coauthors)
        for coauthor in list_coauthors:
            coauthor_name = coauthor.text
            for article in list_articles:
                article = article.contents[0]
                list_authors = [i.text for i in article.find_all('author')]
                if coauthor_name in list_authors:
                    coauthor_names += (coauthor_name + '#')
                    coauthor_years += (article.year.text + '#')
        # citations_nums
        citations_nums = None
        try:
            print('add '.encode('utf-8') + author_name.encode('utf-8'))
        except Exception as e:
            print('print add author_name error')
        author1 = Author(author_name, urlpt, begin_time, article_num, article_keys, article_years, coauthor_num,
                         coauthor_names, coauthor_years, citations_nums)
        try:
            if session.query(Author).filter_by(urlpt = author1.urlpt).first() is None:
                session.add(author1)
            session.commit()
        except Exception as e:
            print('session.commit error')
            session = Session()
            return None
        return author1
    else:
        return author


def get_author_from_sql(index, session):
    """
    从数据库中获取author信息，并处理citations_nums
    return AuthorFormatted
    """
    if index is None:
        return None
    author = session.query(Author).filter_by(id=index).first()

    return author_format(author, session)


def main():
    start = time.time()

    for i in WWW_NUMS:  # 81-200
        print('i = %r' % i)
        filepath = '../../www/' + str(i) + '.xml'
        with open(filepath, 'r') as f:
            content = f.read()
        soup = BeautifulSoup(content, 'lxml')
        list_authors = soup.find_all('www')

        for author in list_authors:  # www标签中的内容 和 api中的内容
            if author.find('author') is not None:  # 解决有的www内没有author问题
                author_name = author.author.text
            else:
                continue
            print(author_name.encode('utf-8'))
            get_author(author_name, session)
            end = time.time()
            print('time = %r' % (end - start))


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    main()
