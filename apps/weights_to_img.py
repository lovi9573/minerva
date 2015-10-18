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

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Use: "+sys.argv[0]+" <datafile> <visible units> <hidden units>"
        sys.quit()
    with open(sys.argv[1], "r") as fin:
        dat = fin.read()
    s_v = int(sys.argv[2])
    s_h = int(sys.argv[3])
    d_x = int(math.sqrt(s_v))
    d_y = s_v/d_x
    
    #Read data into flat array
    dat = re.sub("\s+",",",dat).strip(",")
    dat = dat.split(",")
    dat = map(float,dat)
    dat = np.asarray(dat,dtype = np.float32)
    print dat.shape
    minval = np.min(dat)
    maxval = np.max(dat)
    print "max: {} min: {}\n".format(maxval,minval)
    
    dat = dat.reshape([s_v, s_h])
    #plt.imshow(im0.reshape([d_x,d_y]),interpolation='none')
   # plt.set_cmap('gray')
    #plt.show()
    plt.hist(dat, 10)
    plt.show()
    dat = np.transpose(dat)
    n_horizontal_img = int(math.sqrt(s_h))
    n_vertical_img = s_h/n_horizontal_img
    imgs = np.ndarray([n_vertical_img*(d_y+1),n_horizontal_img*(d_x+1)],dtype=np.float32)
    imgs[:,:] = 0.0
    for img_row in range(n_vertical_img):
        for px_row in range(d_y):
            for img in range(n_horizontal_img):
                imgs[px_row+img_row*(d_y+1),img*(d_x+1):(img+1)*(d_x+1)-1] = dat[img + img_row*n_horizontal_img , px_row*d_x:(px_row+1)*d_x ]
                imgs[:,(img+1)*(d_x+1)-1] = minval
        imgs[px_row+img_row*(d_y+1)+1,:] = minval
                
    plt.imshow(imgs,interpolation='none')
    plt.set_cmap('gray')
    plt.show()
    
#     #dat = np.transpose(dat)
#     for h in range(s_h):
#         im = dat[:,h].reshape([d_y, -1])
#         plt.imshow(im)
#         plt.set_cmap('gray')
#         plt.show()