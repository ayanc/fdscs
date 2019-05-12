#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import pproc as pp
from skimage.io import imread,imsave

import time
import sys
import os

import glob

if len(sys.argv) < 3:
    print("Usage: ./run.py wfile data_dir")
    sys.exit(128)
    
wfile = sys.argv[1]

####
lstr=sys.argv[2] + '/left/%s'
rstr=sys.argv[2] + '/right/%s'
ostr=sys.argv[2] + '/est/%s'

flist = glob.glob(sys.argv[2]+'/left/*_10.png')
flist = [f.split('/')[-1] for f in flist]

model = pp.Net()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model.load(wfile,sess)

for f in flist:
    iml,imr = imread(lstr%f), imread(rstr%f)
    tic=time.time()
    out = model.predict(iml,imr,sess)
    tic=time.time()-tic

    st = "Time: %.3f [%d x %d] " % (tic,out.shape[0],out.shape[1])
    print(st)
    out = np.maximum(1.0,np.minimum(255.0,out)*256.0)
    out = np.uint16(out)
    imsave(ostr%f,out)
    
