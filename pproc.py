#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import slib
import utils as ut
import model
model.toTest()

def lowrescv(limg,rimg,dmax=128):
    shp = tf.shape(limg)[1:]
    ht, wd = shp[0], shp[1]
    ys, xs = tf.mod(ht,2), tf.mod(wd,2)
        
    limg = tf.pad(limg,tf.stack([[0,0],tf.stack([ys,0]),tf.stack([xs,0]),[0,0]]),'SYMMETRIC')
    limg = tf.nn.avg_pool(limg,[1,2,2,1],[1,2,2,1],'VALID')

    rimg = tf.pad(rimg,tf.stack([[0,0],tf.stack([ys,0]),tf.stack([xs,0]),[0,0]]),'SYMMETRIC')
    rimg = tf.nn.avg_pool(rimg,[1,2,2,1],[1,2,2,1],'VALID')

    lrl = limg
    limg, rimg = tf.image.rgb_to_yuv(limg), tf.image.rgb_to_yuv(rimg)
    limg, rimg = slib.census(limg), slib.census(rimg)
        
    cv = slib.hamming(limg,rimg,0,0,(shp[0]+1)//2,(shp[1]+1)//2,dmax)
    cv = (cv-[11.08282948,0.02175535,0.02679042])*[0.1949711,35.91432953,26.79782867]
    shp = tf.shape(cv)
    cv = tf.reshape(cv,tf.stack([shp[0],shp[1],shp[2],-1]))
    return cv,lrl

class Net:
    def graph(self):
        self.left = tf.placeholder(tf.uint8)
        self.right = tf.placeholder(tf.uint8)
        self.gt = tf.placeholder(tf.string)

        limg = tf.expand_dims(tf.to_float(self.left)/255.0,0)
        rimg = tf.expand_dims(tf.to_float(self.right)/255.0,0)
        
        cv,lrl = lowrescv(limg,rimg)
        lrl = 2.0*lrl-1.0
        limg = 2.0*limg-1.0

        self.net = model.Net()
        dout = self.net.predict(limg,cv,lrl)
        self.dout = tf.squeeze(dout,0)

    def load(self,wfile,sess):
        ut.loadNet(wfile,self.net,sess)

    def predict(self,left,right,sess):
        return sess.run(self.dout,feed_dict={self.left: left, self.right: right})

    def __init__(self):
        self.graph()
