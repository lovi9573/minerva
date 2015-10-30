import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import math

"""
Input file must be a flat vector of image values.
Assume n images of v pixels.
All n pixel 0's will be consecutive, followed by all n pixel 1's, etc...

"""

def closeCommonFactor(numbera, numberb, target):
    target = int(target)
    d = 0
    while target+d < numbera :
        if  numbera%(target+d) == 0 and (target+d)%numberb == 0:
            return target+d
        d += 1
    d = 0
    while d < target:
        if numbera%(target-d) == 0 and (target+d)%numberb == 0:
            return target-d
        d += 1
        

if __name__ == "__main__":
    isprob = False
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print "Use: "+sys.argv[0]+" <datafile> [-p] \n\t-p  treat values as probabilities (0,1)"
        sys.exit()
    if len(sys.argv) == 3 and sys.argv[2] == "-p":
        isprob=True
    with open(sys.argv[1], "r") as fin:
        header = fin.readline()
        dat = fin.read()
        
    d_x,d_y,c,n = map(int,header.strip().split(" ")) 
    
    #Read data into flat array
    dat = re.sub("\s+",",",dat).strip(",")
    dat = dat.split(",")
    dat = map(float,dat)
    dat = np.asarray(dat,dtype = np.float32)
    minval = np.min(dat)
    maxval = np.max(dat)
    print "max: {} min: {}\n".format(maxval,minval)
    
    n_h_ideal = math.sqrt(n*c)
    n_h = closeCommonFactor(n*c, c, n_h_ideal)
    n_v = n*c/n_h
    print n_h, n_v
    dat = dat.reshape([n_v,n_h,d_y,d_x])
    #plt.imshow(im0.reshape([d_x,d_y]),interpolation='none')
   # plt.set_cmap('gray')
    #plt.show()
    #plt.hist(dat.flatten(), 10)
    #plt.show()
    
    #dat = np.transpose(dat)
    imgs = np.ndarray([n_v*(d_y+1),n_h*(d_x+1)],dtype=np.float32)
    imgs[:,:] = 0.0
    for img_row in range(n_v):
        for px_row in range(d_y):
            for img in range(n_h):
                imgs[px_row+img_row*(d_y+1),img*(d_x+1):(img+1)*(d_x+1)-1] = dat[img_row, img, px_row , :]
                imgs[:,(img+1)*(d_x+1)-1] = minval
        imgs[px_row+img_row*(d_y+1)+1,:] = minval
    if isprob:
        plt.imshow(imgs,interpolation='none',vmin=0,vmax=1)   
    else:          
        plt.imshow(imgs,interpolation='none')
    plt.set_cmap('gray')
    plt.show()
    
