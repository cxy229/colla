GRAPH_PATH = './graph/'

from mergecolla import *
import networkx as nx
G = nx.Graph()

def generate_graph(engine_all, session_all):
    start_time = time.time()
    with engine_all.connect() as con:

        meta = MetaData(engine_all)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla])
        rs = con.execute(stm)
        for tmp in rs:
            G.add_edge(tmp[1], tmp[2], year = tmp[3])
    # nx.write_multiline_adjlist(G, GRAPH_PATH + 'all.txt')
    # print(GRAPH_PATH + 'all.txt')
    print('graph time = %r' %(time.time() - start_time))
    min_year = 1950
    for max_year in [2016-i for i in range(1,2016-min_year+1)]:
        # 生成max_year 前的图
        g_tmp = G.copy()
        for edge in g_tmp.edges_iter(data='year', default=-1):
            if edge[2] >= max_year:
                G.remove_edge(edge[0], edge[1])
        # nx.write_multiline_adjlist(G, GRAPH_PATH + str(max_year) +'.txt')
        # print(GRAPH_PATH + str(max_year) +'.txt')
        collas = session_all.query(Colla).filter_by(begin_time = max_year).all()
        for colla in collas:
            if G.has_node(colla.author_name1) and G.has_node(colla.author_name2):
                # colla.betweenness_centrality1 = nx.betweenness_centrality(g_tmp)[colla.author_name1.encode('utf-8')]
                # colla.betweenness_centrality2 = nx.betweenness_centrality(g_tmp)[colla.author_name2.encode('utf-8')]
                # print(colla.betweenness_centrality1, colla.betweenness_centrality2)
                if nx.has_path(G, colla.author_name1, colla.author_name2):
                    print(colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8'))
                    colla.shortest_path_length = nx.shortest_path_length(G, colla.author_name1, colla.author_name2)
                    print(colla.shortest_path_length)
        session_all.commit()
        print('graph %r time = %r' %max_year, (time.time() - start_time))

def load_graph(year):
    with open(GRAPH_PATH+str(year)+'.txt', 'rb') as f:
        return nx.read_multiline_adjlist(f)

def shortest_path_length(session_all):
    for max_year in range(1950, 1951):
        g_tmp = load_graph(max_year)
        collas = session_all.query(Colla).filter_by(begin_time = max_year).all()
        for colla in collas:
            if g_tmp.has_node(colla.author_name1.encode('utf-8')) and g_tmp.has_node(colla.author_name2.encode('utf-8')):
                # colla.betweenness_centrality1 = nx.betweenness_centrality(g_tmp)[colla.author_name1.encode('utf-8')]
                # colla.betweenness_centrality2 = nx.betweenness_centrality(g_tmp)[colla.author_name2.encode('utf-8')]
                # print(colla.betweenness_centrality1, colla.betweenness_centrality2)
                if nx.has_path(g_tmp, colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8')):
                    print(colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8'))
                    colla.shortest_path_length = nx.shortest_path_length(g_tmp, colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8'))
                    print(colla.shortest_path_length)
        session_all.commit()

def betweenness_centrality(engine_all, session_all):
    start_time = time.time()
    with engine_all.connect() as con:

        meta = MetaData(engine_all)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla])
        rs = con.execute(stm)
        for tmp in rs:
            G.add_edge(tmp[1], tmp[2], year = tmp[3])
    print('graph time = %r' %(time.time() - start_time))
    min_year = 2016
    for max_year in [2016-i for i in range(0,2016-min_year+1)]:
        # 生成max_year 前的图
        g_tmp = G.copy()
        for edge in g_tmp.edges_iter(data='year', default=-1):
            if edge[2] >= max_year:
                G.remove_edge(edge[0], edge[1])
        collas = session_all.query(Colla).filter_by(begin_time = max_year).all()
        for colla in collas:
            if G.has_node(colla.author_name1) and G.has_node(colla.author_name2):
                # colla.betweenness_centrality1 = nx.betweenness_centrality(G)[colla.author_name1]
                # colla.betweenness_centrality2 = nx.betweenness_centrality(G)[colla.author_name2]
                # print(colla.betweenness_centrality1, colla.betweenness_centrality2)
                betweenness_centrality1 = nx.betweenness_centrality(G)[colla.author_name1]
                betweenness_centrality2 = nx.betweenness_centrality(G)[colla.author_name2]
                print(betweenness_centrality1, betweenness_centrality2)
        print('graph %r time = %r' %max_year, (time.time() - start_time))

# TODO:
def main():
    start_time = time.time()
    # 生成图

    print('graph time = %r' %(time.time() - start_time))
    for colla_index in range(1, 10):
        colla = session_colla.query(Colla).filter_by(id=colla_index).first()
        max_year = colla.begin_time
        g_tmp = nx.Graph()
        # 生成max_year 前的图
        for edge in G.edges_iter(data='year', default=-1):
            if edge[2] < max_year:
                g_tmp.add_edge(edge[0], edge[1], year = edge[2])
        print('max_year time = %r' %(time.time() - start_time))

        if g_tmp.has_node(colla.author_name1.encode('utf-8')) and g_tmp.has_node(colla.author_name2.encode('utf-8')):
            colla.betweenness_centrality1 = nx.betweenness_centrality(g_tmp)[colla.author_name1.encode('utf-8')]
            colla.betweenness_centrality2 = nx.betweenness_centrality(g_tmp)[colla.author_name2.encode('utf-8')]
            print(colla.betweenness_centrality1, colla.betweenness_centrality2)
            if nx.has_path(g_tmp, colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8')):
                print(colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8'))
                colla.shortest_path_length = nx.shortest_path_length(g_tmp, colla.author_name1.encode('utf-8'), colla.author_name2.encode('utf-8'))
                print(colla.shortest_path_length)

        # print(nx.betweenness_centrality(g_tmp))
        session_colla.commit()

if __name__ == '__main__':
    # generate_graph(engine_all, session_all)
    # shortest_path_length(session_all)
    betweenness_centrality(engine_all, session_all)
