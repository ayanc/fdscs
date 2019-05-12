#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import slib

def lowrescv(limg,rimg,imsz,dmax=128):
        #shp = tf.shape(limg)[1:]
        #ht, wd = shp[0], shp[1]
        #ys, xs = tf.mod(ht,2), tf.mod(wd,2)
        
        #limg = tf.pad(limg,tf.stack([[0,0],tf.stack([ys,0]),tf.stack([xs,0]),[0,0]]),'SYMMETRIC')
        limg = tf.nn.avg_pool(limg,[1,2,2,1],[1,2,2,1],'VALID')

        #rimg = tf.pad(rimg,tf.stack([[0,0],tf.stack([ys,0]),tf.stack([xs,0]),[0,0]]),'SYMMETRIC')
        rimg = tf.nn.avg_pool(rimg,[1,2,2,1],[1,2,2,1],'VALID')

        lim0 = limg
        limg, rimg = tf.image.rgb_to_yuv(limg), tf.image.rgb_to_yuv(rimg)
        limg, rimg = slib.census(limg), slib.census(rimg)
        
        cv = slib.hamming(limg,rimg,0,0,imsz[0]//2,imsz[1]//2,dmax)
        cv = (cv-[11.08282948,0.02175535,0.02679042])*[0.1949711,35.91432953,26.79782867]

        shp = tf.shape(cv)
        cv = tf.reshape(cv,tf.stack([shp[0],shp[1],shp[2],-1]))
        return cv,lim0

class dataset:
    def graph(self):
        self.left = []
        self.right = []
        self.gtl = []
        self.gtr = []

        # Create placeholders for image names
        for i in range(self.bsz):
            self.left.append(tf.placeholder(tf.string))
            self.right.append(tf.placeholder(tf.string))
            self.gtl.append(tf.placeholder(tf.string))
            self.gtr.append(tf.placeholder(tf.string))

        # Load Images
        limgs, rimgs, gts = [], [], []
        imsz = self.imsz

        for i in range(self.bsz):
            limg = tf.image.decode_jpeg(tf.read_file(self.left[i]),channels=3)
            rimg = tf.image.decode_jpeg(tf.read_file(self.right[i]),channels=3)
            gtl = tf.image.decode_png(tf.read_file(self.gtl[i]),channels=1,dtype=tf.uint16)
            gtr = tf.image.decode_png(tf.read_file(self.gtr[i]),channels=1,dtype=tf.uint16)

            limgs.append(limg)
            limgs.append(rimg[:,::-1,:])

            rimgs.append(rimg)
            rimgs.append(limg[:,::-1,:])
            gts.append(gtl)
            gts.append(gtr[:,::-1,:])

        # Stack & Convert
        limgs = tf.to_float(tf.stack(limgs))/255.0
        rimgs = tf.to_float(tf.stack(rimgs))/255.0
        gts = tf.to_float(tf.stack(gts))/256.0
        mask = tf.to_float(gts < np.float32(255.0))

        self.cv, lrl = lowrescv(limgs,rimgs,self.imsz)
        self.lrl = 2.0*lrl-1.0
        self.limgs = 2.0*limgs-1.0
        self.disp = gts
        self.mask = mask

    def fdict(self,ids):
        fd = {}
        assert len(ids) == self.bsz
        for i in range(self.bsz):
            fd[self.left[i]] = ids[i] % self.ligrp
            fd[self.right[i]] = ids[i] % self.rigrp
            fd[self.gtl[i]] = ids[i] % self.lggrp
            fd[self.gtr[i]] = ids[i] % self.rggrp
            
        return fd

    '''
    bsz = Batch Size
    csz = Crop Size (of final depth map output)
    rsz = Receptive Field of NN (to pad image)
    nsz = Filter Size for Smoothing (to pad cost volume)

    Run fetchOp to get lrgb, disp, mask, cv
    '''
    def __init__(self,bsz, \
                 ligrp=('frames_finalpass','left','webp.jpg'), \
                 rigrp=('frames_finalpass','right','webp.jpg'), \
                 lggrp=('disparity','left','png'), \
                 rggrp=('disparity','right','png'), \
                 imsz=[540,960]):

        self.bsz = bsz
        self.ligrp,self.rigrp = ligrp, rigrp
        self.lggrp,self.rggrp = lggrp, rggrp
        self.imsz = imsz

        self.graph()
