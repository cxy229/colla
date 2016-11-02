import os
path = './result/2/80/time/'
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file)) and file!='.DS_Store':
        os.rename(os.path.join(path, file),os.path.join(path, file[:-1]))
        print(path+file[:-1])
