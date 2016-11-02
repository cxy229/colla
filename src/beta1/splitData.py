from sqlalchemy import MetaData, Table, select
from collamodel import engine, session, Colla_20,Colla_40,Colla_60,Colla_80


if __name__ == '__main__':
    # 遍历
    with engine.connect() as con:
        meta = MetaData(engine)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla])   # .where(colla.c.id < 100)
        rs = con.execute(stm)
        for tmp in rs:
            if tmp[21]>20 and tmp[22]>20:
                print(tmp[0], tmp[1], tmp[2],tmp[21],tmp[22])
                t = Colla_20(tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9],tmp[10],tmp[11],tmp[12],tmp[13],
                         tmp[14],tmp[15],tmp[16],tmp[17],tmp[18],tmp[19],tmp[20],tmp[21],tmp[22])
                session.add(t)
                if tmp[21] > 40 and tmp[22] > 40:
                    print(tmp[0], tmp[1], tmp[2], tmp[21], tmp[22])
                    t = Colla_40(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10],
                                 tmp[11], tmp[12], tmp[13],
                                 tmp[14], tmp[15], tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22])
                    session.add(t)
                    if tmp[21] > 60 and tmp[22] > 60:
                        print(tmp[0], tmp[1], tmp[2], tmp[21], tmp[22])
                        t = Colla_60(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10],
                                     tmp[11], tmp[12], tmp[13],
                                     tmp[14], tmp[15], tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22])
                        session.add(t)
                        if tmp[21] > 80 and tmp[22] > 80:
                            print(tmp[0], tmp[1], tmp[2], tmp[21], tmp[22])
                            t = Colla_80(tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9],
                                         tmp[10], tmp[11], tmp[12], tmp[13],
                                         tmp[14], tmp[15], tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21],
                                         tmp[22])
                            session.add(t)
        session.commit()
