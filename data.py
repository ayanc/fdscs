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
        self.gt = []

        # Create placeholders for image names
        for i in range(self.bsz):
            self.left.append(tf.placeholder(tf.string))
            self.right.append(tf.placeholder(tf.string))
            self.gt.append(tf.placeholder(tf.string))

        # Load Images
        limgs, rimgs, gts = [], [], []
        imsz = self.imsz
        crshp = tf.constant([imsz[0],imsz[1],-1],dtype=tf.int32)

        for i in range(self.bsz):
            limg = tf.image.decode_jpeg(tf.read_file(self.left[i]),channels=3)
            rimg = tf.image.decode_jpeg(tf.read_file(self.right[i]),channels=3)
            gt = tf.image.decode_png(tf.read_file(self.gt[i]),channels=1,dtype=tf.uint16)

            scale = tf.random_uniform([], 1.0, 1.5, dtype=tf.float32)
            yscale = tf.random_uniform([], 1.0, 1.5, dtype=tf.float32)
            oshp = tf.to_float(tf.shape(limg))
            oshp = tf.to_int32(tf.stack([yscale*oshp[0],scale*oshp[1]]))
        
            limg = tf.image.resize_images(limg,oshp,tf.image.ResizeMethod.BILINEAR)
            rimg = tf.image.resize_images(rimg,oshp,tf.image.ResizeMethod.BILINEAR)
            gt = tf.image.resize_images(gt,oshp,tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            imshp = oshp
            yc = tf.random_uniform((),0,imshp[0]-imsz[0]+1,dtype=tf.int32)
            xc = tf.random_uniform((),0,imshp[1]-imsz[1]+1,dtype=tf.int32)
        
            limg = tf.slice(limg,tf.stack([yc,xc,0]),crshp)
            rimg = tf.slice(rimg,tf.stack([yc,xc,0]),crshp)
            gt = tf.slice(gt,tf.stack([yc,xc,0]),crshp)

            gt = tf.to_float(gt)*scale/256.0

            limgs.append(limg)
            rimgs.append(rimg)

            gts.append(gt)

        # Stack & Convert
        limgs = tf.to_float(tf.stack(limgs))/255.0
        rimgs = tf.to_float(tf.stack(rimgs))/255.0
        gts = tf.stack(gts)
        mask = tf.to_float(gts > 0)*tf.to_float(gts < 255.0)

        self.cv, lrl = lowrescv(limgs,rimgs,self.imsz)
        self.lrl = 2.0*lrl-1.0
        self.limgs = 2.0*limgs-1.0
        self.disp = gts
        self.mask = mask

    def fdict(self,ids):
        fd = {}
        assert len(ids) == self.bsz
        for i in range(self.bsz):
            fd[self.left[i]] = self.lstr % ids[i]
            fd[self.right[i]] = self.rstr % ids[i]
            fd[self.gt[i]] = self.dstr % ids[i]
            
        return fd

    def __init__(self,bsz, \
                 lstr='data/ktboth/left/%s', \
                 rstr='data/ktboth/right/%s', \
                 dstr='data/ktboth/disp/%s',
                 imsz=[370,1224]):

        self.bsz = bsz
        self.lstr,self.rstr,self.dstr = lstr, rstr, dstr
        self.imsz = imsz

        self.graph()
