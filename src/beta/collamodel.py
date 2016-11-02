# -*- coding: utf-8 -*-

start_index = 175000
end_index = 180000
# colla_sqlname = 90000

import time
from sqlalchemy import create_engine, func
from sqlalchemy import MetaData, Table, select

# TODO:
engine_author = create_engine('sqlite:///../../sql/author.sqlite', echo=False)
# engine_colla = create_engine('sqlite:///../../sql/colla'+ str(colla_sqlname) +'.sqlite', echo=False)
engine_colla = create_engine('mysql+pymysql://root:2222@localhost/colla', echo=False)

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker

def split_sharp(s):
    """
     分割字符串
    Args:
        s:

    Returns:

    """


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


class Author(Base):
     __tablename__ = 'Author'
     id = Column(Integer, primary_key=True)
     name = Column(String(50), unique=True)
     urlpt = Column(String(50), unique=True)
     begin_time = Column(Integer, default=2222)
     article_num = Column(Integer, default=0)
     article_keys = Column(Text, default=None)
     article_years = Column(Text, default=None)
     coauthor_num = Column(Integer, default=0)
     coauthor_names = Column(Text, default=None)
     coauthor_years = Column(Text, default=None)
     citations_nums = Column(Text, default=None)

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
    colla_time = Column(Integer, default=-1)

    coarticle_num = Column(Integer, default=-1)
    coarticle_years = Column(Text, default=None)

    scientific_age1 = Column(Integer, default=-1)
    scientific_age2 = Column(Integer, default=-1)
    article_num1 = Column(Integer, default=-1)
    article_num2 = Column(Integer, default=-1)
    citation_num1 = Column(Integer, default=-1)
    citation_num2 = Column(Integer, default=-1)
    common_neighbors_num = Column(Integer, default=-1)
    shortest_path_length = Column(Integer, default=2222)

    degree1 = Column(Integer, default=-1)
    degree2 = Column(Integer, default=-1)

    betweenness_centrality1 = Column(Integer, default=-1)
    betweenness_centrality2 = Column(Integer, default=-1)

    scientific_age_to2016_1 = Column(Integer, default=-1)
    scientific_age_to2016_2 = Column(Integer, default=-1)

    article_num_to2016_1 = Column(Integer, default=-1)
    article_num_to2016_2 = Column(Integer, default=-1)


    def __init__(self, author_name1, author_name2, begin_time, colla_time, coarticle_num, coarticle_years,
                 scientific_age1, scientific_age2, article_num1, article_num2, citation_num1, citation_num2,
                 common_neighbors_num, shortest_path_length, degree1, degree2, betweenness_centrality1, betweenness_centrality2,
                 scientific_age_to2016_1, scientific_age_to2016_2, article_num_to2016_1, article_num_to2016_2):
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

        self.degree1 = degree1
        self.degree2 = degree2
        self.betweenness_centrality1 = betweenness_centrality1
        self.betweenness_centrality2 = betweenness_centrality2
        self.scientific_age_to2016_1 = scientific_age_to2016_1
        self.scientific_age_to2016_2 = scientific_age_to2016_2
        self.article_num_to2016_1 = article_num_to2016_1
        self.article_num_to2016_2 = article_num_to2016_2


    def __repr__(self):
        return "<Colla('%s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)

class AuthorFormatted(object):
    '''
    class Author format
    '''
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

Session_Author = sessionmaker(bind=engine_author)
session_author = Session_Author()

Base.metadata.create_all(bind=engine_colla)
Session_Colla = sessionmaker(bind=engine_colla)
session_colla = Session_Colla()

def author_format(author):
    '''
    把author格式化
    Returns:
        AuthorFormatted
        None:
    '''
    if type(author) == Author:
        coauthor_names = split_sharp(author.coauthor_names)
        coauthor_years = split_sharp(author.coauthor_years)
        coauthor_years = [int(year) for year in coauthor_years]   # year 转换成 int

        article_keys = split_sharp(author.article_keys)
        article_years = split_sharp(author.article_years)
        article_years = [int(year) for year in article_years]   # year 转换成 int

        citations_nums = None

        return AuthorFormatted(author.name, author.urlpt, author.begin_time, author.article_num, article_keys, article_years, author.coauthor_num, coauthor_names, coauthor_years, citations_nums)
    else:
        return None


def get_author_from_sql(index, session):
    '''
    从数据库中获取author信息，并处理citations_nums
    return AuthorFormatted
    '''
    if index is None:
        return None
    if type(index) is int:
        author = session.query(Author).filter_by(id = index).first()
    elif type(index) is str:
        author = session.query(Author).filter_by(name = index).first()

    return author_format(author)


def traversal_coauthor(author_formatted, session_author, session, start_time):
    """遍历coauthor，生成Colla，加入数据库
    Args:
        author_formatted:
        session_author:
        session: session_colla
        start_time:
    Returns:
        None: args 为None
    """
    if author_formatted is None:
        return None
    author_name1 = author_formatted.name
    pre_coauthor = ''
    coarticle_years = ''
    list_coarticle_years = []
    for coauthor_index in range(0,len(author_formatted.coauthor_names)):    # 遍历 coauthor
        end_time = time.time()
        if author_formatted.coauthor_names[coauthor_index] == pre_coauthor: #
            list_coarticle_years.append(author_formatted.coauthor_years[coauthor_index])
            continue
        # 新 coauthor
        # 处理 旧coauthor
        if pre_coauthor == '':
            pre_coauthor = author_formatted.coauthor_names[coauthor_index]
            list_coarticle_years.append(author_formatted.coauthor_years[coauthor_index])
        else:
            list_coarticle_years.sort()
            begin_time = list_coarticle_years[0]      # get begin_time
            colla_time = list_coarticle_years[-1] - begin_time  # get colla_time
            coarticle_num = len(list_coarticle_years)   # get coarticle_num
            for year in list_coarticle_years:   # get coarticle_years
                coarticle_years += (str(year)+'#')
            # tmp_author = session.query(Colla).filter_by(author_name1=pre_coauthor).first() # 解决重复的author
            # if tmp_author is None:
            scientific_age1 = begin_time - author_formatted.begin_time  # get scientific_age1
            author2_formatted = get_author_from_sql(pre_coauthor, session_author)
            if author2_formatted:
                author_name2 = author2_formatted.name
                scientific_age2 = begin_time - author2_formatted.begin_time # get scientific_age2
                article_num1 = len([year for year in author_formatted.article_years if year<begin_time])    # get article_num1
                article_num2 = len([year for year in author2_formatted.article_years if year<begin_time])   # get article_num2
                                                                    # get common_neighbors_num
                coauthor_names1 =  author_formatted.coauthor_names # coauthor_names1
                coauthor_years1 = author_formatted.coauthor_years
                coauthor_names2 = author2_formatted.coauthor_names  # coauthor_names2
                coauthor_years2 = author2_formatted.coauthor_years
                article_years1 = author_formatted.article_years
                article_years2 = author2_formatted.article_years

                common_neighbors = set()
                for k in range(0,len(coauthor_names1)):
                    if coauthor_names1[k] in coauthor_names2 and int(coauthor_years1[k]) <= begin_time:
                        if int(coauthor_years2[coauthor_names2.index(coauthor_names1[k])]) <= begin_time:
                            common_neighbors.add(coauthor_names1[k])
                common_neighbors_num = len(common_neighbors)

                citations_nums1 = -1   # get citation_num1

                                       # get citation_num2
                citations_nums2 = -1

                shortest_path_length = 2222
                betweenness_centrality1 = -1
                betweenness_centrality2 = -1

                # get degree
                coauthor1 = set()
                for i in range(0, len(coauthor_names1)):
                    if coauthor_years1[i] < begin_time:
                        coauthor1.add(coauthor_names1[i])
                degree1 = len(coauthor1)

                coauthor2 = set()
                for i in range(0, len(coauthor_names2)):
                    if coauthor_years2[i] < begin_time:
                        coauthor2.add(coauthor_names2[i])
                degree2 = len(coauthor2)

                # get scientific_age_to2016
                tmp = article_years1.copy()
                tmp.sort()
                scientific_age_to2016_1 = tmp[-1] - tmp[0]
                article_num_to2016_1 = author_formatted.article_num
                tmp = article_years2.copy()
                tmp.sort()
                scientific_age_to2016_2 = tmp[-1] - tmp[0]
                article_num_to2016_2 = author2_formatted.article_num

                citation_num1 = -1
                citation_num2 = -1

                # if session.query(Colla).filter_by(author_name1 = author_name1, author_name2 = author_name2).first() is None:
                colla = Colla(author_name1, author_name2, begin_time, colla_time, coarticle_num, coarticle_years,
                             scientific_age1, scientific_age2, article_num1, article_num2, citation_num1, citation_num2,
                             common_neighbors_num, shortest_path_length, degree1, degree2, betweenness_centrality1, betweenness_centrality2,
                             scientific_age_to2016_1, scientific_age_to2016_2, article_num_to2016_1, article_num_to2016_2)
                session.add(colla)
                print(author_name1.encode('utf-8'),author_name2.encode('utf-8'))
            # 更新
            pre_coauthor = author_formatted.coauthor_names[coauthor_index]
            list_coarticle_years = []
            list_coarticle_years.append(author_formatted.coauthor_years[coauthor_index])
            coarticle_years = ''
    session.commit()
    print('time = %r' %(end_time-start_time))

def author2author(session_author,session_colla):
    with engine_author.connect() as con:
        meta = MetaData(engine_author)
        author = Table('Author', meta, autoload=True)
        stm = select([author]) # .where(author.c.id == 2045)
        rs = con.execute(stm)
        for tmp in rs:
            print(tmp[0],tmp[1])
            author = Author(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], None)
            session_colla.add(author)
        session_colla.commit()

def main():
    start_time = time.time()
    # Base.metadata.create_all(engine)
    for j in range(start_index, end_index): # ?98 ?301  遍历 Author表
        print('j = %r' %j)
        author = get_author_from_sql(j, session_author)

        if len(author.coauthor_names) < 1:
            continue

        traversal_coauthor(author, session_author, session_colla, start_time)

if __name__ == '__main__':
    # main()
    author2author(session_author,session_colla)