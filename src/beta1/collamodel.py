from sqlalchemy import create_engine, func, Table, MetaData
from sqlalchemy.sql import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker

engine = create_engine('mysql+pymysql://root:2222@localhost/colla', echo=False)

Base = declarative_base()


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
        return "<Colla(' %s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)


class Colla_20(Base):
    '''
     sqlalchemy Colla table
     '''
    __tablename__ = 'Colla_20'
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
        return "<Colla(' %s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)


class Colla_40(Base):
    '''
     sqlalchemy Colla table
     '''
    __tablename__ = 'Colla_40'
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
        return "<Colla(' %s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)

class Colla_60(Base):
    '''
     sqlalchemy Colla table
     '''
    __tablename__ = 'Colla_60'
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
        return "<Colla(' %s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)

class Colla_80(Base):
    '''
     sqlalchemy Colla table
     '''
    __tablename__ = 'Colla_80'
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
        return "<Colla(' %s','%s', '%d', '%d')>" % (
        self.author_name1, self.author_name2, self.begin_time, self.colla_time)



Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()
