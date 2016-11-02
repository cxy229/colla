import re

# 分割文件
f = open('./dblp.xml', 'r')
# f = open('./example.xml', 'r')
chunk_file = open('./data/1.xml', 'w')
count = 1
chunk_num = 0
flag = False     # 提取<www>
while True:
    line = f.readline()
    if line == '':
        break
    if re.search(r'<www', line) is not None:
        # print('flag become false')
        flag = True
    if flag:
        if line == '</www>\n':
            chunk_file.write(line)
            chunk_num += 1
            if chunk_num == 100:    # 一个文件放100个article
                chunk_num = 0
                chunk_file.close()
                count += 1
                file_path = './www/'+str(count)+'.xml'
                chunk_file = open(file_path, 'w')
                print('count=%r' %count)
                continue
        chunk_file.write(line)
    if re.search(r'</www', line) is not None:
        flag = False
