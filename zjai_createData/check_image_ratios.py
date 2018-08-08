import os.path as osp
from  zjai_createData.check_exist import get_all_file
import PIL.Image

def get_all_ratios(fileList):
    max=0
    maxname=''
    min=1000
    minname=''
    count=0
    for filename in fileList:
        ratios=open_image_by_PIL(filename)
        if max<ratios:
            max=ratios
            maxname=filename
        if min>ratios:
            min=ratios
            minname=filename
        if count%10000==0:
            print(count,max,min)
        if ratios>15 or ratios<(1/15):
            print(maxname,ratios)
        count+=1
    print("{} is max:{}".format(maxname,max))
    print("{} is min:{}".format(minname, min))

def open_image_by_PIL(filename):
    width,height=PIL.Image.open(filename).size
    ratios=width/height
    return ratios

if __name__=="__main__":
    root_dir=osp.abspath(osp.join(osp.dirname(__file__),".."))
    data_dir=osp.join(root_dir,"data","train_data")
    fileList=get_all_file(data_dir,"jpg")
    get_all_ratios(fileList)
