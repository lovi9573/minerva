import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import math


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Use: "+sys.argv[0]+" <datafile> <visible units> <hidden units>"
        sys.quit()
    with open(sys.argv[1], "r") as fin:
        dat = fin.read()
    s_v = int(sys.argv[2])
    s_h = int(sys.argv[3])
    dat = re.sub("\s+",",",dat).strip(",")
    print dat
    dat = dat.split(",")
    dat = map(float,dat)
    print dat
    dat = np.asarray(dat,dtype = np.float32)
    print dat.shape
    
    dat = dat.reshape([s_h,s_v])
    dat = np.transpose(dat)
    for h in range(s_h):
        im = dat[:,h].reshape([int(math.sqrt(s_v)), -1])
        plt.imshow(im)
        plt.set_cmap('gray')
        plt.show()