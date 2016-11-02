import time
from sqlalchemy import create_engine, func, Table, MetaData
from sqlalchemy.sql import select

engine_part = create_engine('sqlite:///../../sql/1', echo=False)
# engine_colla = create_engine('sqlite:///../../sql/colla'+ str(colla_sqlname) +'.sqlite', echo=False)
engine_all = create_engine('mysql+pymysql://root:2222@localhost/colla', echo=False)

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker

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

Session_Part = sessionmaker(bind=engine_part)
session_part = Session_Part()

Base.metadata.create_all(bind=engine_all)
Session_All = sessionmaker(bind=engine_all)
session_all = Session_All()

def main():
    # part_len = session_part.query(func.max(Colla.id)).scalar()
    # for colla_index in range(0, part_len):
    with engine_part.connect() as con:

        meta = MetaData(engine_part)
        colla = Table('Colla', meta, autoload=True)

        stm = select([colla])
        rs = con.execute(stm)
        for tmp in rs:
            c = Colla(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10], tmp[11]
            , tmp[12], tmp[13], tmp[14], tmp[15], tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22])
            session_all.add(c)
            print(c)
        session_all.commit()

if __name__ == '__main__':
    main()
