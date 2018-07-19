import re
import os
import os.path as osp

def get_size(filename):
    regex='(?<=width>).+?(?=<)'
    regex1 = '(?<=height>).+?(?=<)'
    pattern=re.compile(regex)
    pattern1=re.compile(regex1)
    with open(filename,"r") as f:
        lines=f.read()
        width=int(float(re.findall(pattern,lines)[0]))
        height = int(float(re.findall(pattern1, lines)[0]))
        # print(width,height)
    return width,height


def get_path(dataSet):
    import time
    count=0
    start = time.time()

    resultList=[]
    with open(dataSet+'/trainval_sizes.txt','w') as f:
        for parent,dirname,filenames in os.walk(dataSet):
            for filename in filenames:
                if filename[-3:] == "xml":
                    filename=osp.join(parent,filename)
                    width, height=get_size(filename)
                    resultList.append("{}|{}|{}".format(filename.replace("Annotations","JPEGImages")
                                                        .replace(".xml",".jpg"),str(width),str(height)))

                    count+=1
                    if count%10000==0:
                        print(count,time.time()-start)
                        start=time.time()


        f.write("\n".join(resultList))



if __name__=="__main__":
    root_dir=osp.abspath(osp.join(osp.dirname(__file__),".."))
    root_dir='/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/'
    data_dirs=osp.join(root_dir,'data',"train_data")
    get_path(data_dirs)