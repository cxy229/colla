BEGIN_INDEX = 0
END_INDEX = 100000

from sqlalchemy.sql import select
from sqlalchemy import create_engine, func, Table, MetaData
from collamodel import *

engine_colla_new = create_engine('mysql+pymysql://root:2222@localhost/colla_new', echo=False)
Base.metadata.create_all(bind=engine_colla_new)
Session_Colla_New = sessionmaker(bind=engine_colla_new)
session_colla_new = Session_Colla_New()

def rmRepetition():
    """
    colla 去重
    """
    start_time = time.time()
    with engine_colla.connect() as con:
        meta = MetaData(engine_colla)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla]).where((colla.c.id > BEGIN_INDEX) & (colla.c.id < END_INDEX))
        rs = con.execute(stm)

        count = BEGIN_INDEX
        for tmp in rs:
            if session_colla_new.query(Colla).filter_by(author_name1 = tmp[2], author_name2 = tmp[1]).first() is None:
                c = Colla(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10], tmp[11],
                    tmp[12], tmp[13], tmp[14], tmp[15], tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22])
                # print(c.encode('utf-8'))
                session_colla_new.add(c)
                session_colla_new.commit()
            count += 1
            if count % 100 == 0:
                print(count,'time = %r' %(time.time() - start_time))

if __name__ == '__main__':
    rmRepetition()
