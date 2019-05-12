#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_nn_ops

nfCV = 128*3

bneps = 1e-8
ifbn = 1
numCh = 32
nChInc = 16
nlev = 5
nfinal=3
topad = 2**(nlev-1)

encs = [nfCV,192,96,48,32]

upsamp=False

# Remove BNs (use after replacing filters with popstats)    
def toTest():
    global ifbn, upsamp
    ifbn = 0
    upsamp=True

def conv(net,name,inp,ksz,bn=0,relu=True):
    if len(ksz) > 2:
        inch = ksz[2]
    else:
        inch = int(inp.get_shape()[-1])

    ksz = [ksz[0],ksz[0],inch,ksz[1]]

    wnm = name + "_w"
    if wnm in net.weights.keys():
        w = net.weights[wnm]
    else:
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        w = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        net.weights[wnm] = w
        net.wd = net.wd + tf.nn.l2_loss(w)

    out=tf.nn.conv2d(inp,w,[1,1,1,1],'SAME')

    if bn==1:
        mu,vr = tf.nn.moments(out,[0,1,2])
        net.bnvals[name + '_m'] = mu
        net.bnvals[name + '_v'] = vr
        out = tf.nn.batch_normalization(out,mu,vr,None,None,bneps)
        
    bnm = name + "_b"
    if bnm in net.weights.keys():
        b = net.weights[bnm]
    else:
        b = tf.Variable(tf.constant(0,shape=[ksz[-1]],dtype=tf.float32))
        net.weights[bnm] = b
    out = out+2.0*b

    if relu:
        out = tf.nn.relu(out)

    return out

def unet(net,inp,nch,inch,nlev,pfx='u',_ch=None):

    out = inp

    if _ch is None:
        out = conv(net,pfx+'u1_%d'%nlev,out,[3,nch],0,True)
    else:
        out = conv(net,pfx+'u1_%d'%nlev,out,[3,nch,_ch],0,True)
    out = conv(net,'u2_%d'%nlev,out,[3,nch],0,True)

    if nlev == 1:
        return out

    out1 = tf.nn.max_pool(out,[1,2,2,1],[1,2,2,1],'VALID')
    out1 = unet(net,out1,nch+inch,inch,nlev-1,pfx)
    out1 = conv(net,pfx+'u3_%d'%nlev,out1,[1,nch*4],0,True)
    out1 = tf.depth_to_space(out1,2)

    out = tf.concat([out,out1],3)
    
        
    out = conv(net,'u4_%d'%nlev,out,[3,nch],0,True)
    out = conv(net,pfx+'u5_%d'%nlev,out,[3,nch],0,True)

    return out
    

class Net:
    def __init__(self):
        self.weights = {}
        self.bnvals = {}
        self.wd = 0.
        
    # Encode images to feature tensor
    def predict(self, img, cv, lrl):

        fts = cv
        for i in range(len(encs)-1):
            fts = conv(self,'enc%d'%i,fts,[1,encs[i+1],encs[i]],ifbn)

        fts = tf.concat([lrl,fts],axis=3)
        fts = conv(self,'cenc',fts,[3,encs[-1],3+encs[-1]],ifbn)
        for i in range(nfinal-1):
            fts = conv(self,'cenc%d'%(i+1),fts,[3,encs[-1],encs[-1]],ifbn)
            
        fshp = tf.shape(fts)    
        inp = tf.concat([lrl, fts], axis=3)

        ypad = tf.mod(topad-tf.mod(fshp[1],topad),topad)
        ypad = [ypad//2,ypad-ypad//2]
        xpad = tf.mod(topad-tf.mod(fshp[2],topad),topad)
        xpad = [xpad//2,xpad-xpad//2]
        inp = tf.pad(inp,tf.stack([[0,0],tf.stack(ypad),tf.stack(xpad),[0,0]]),'REFLECT')
        
        out = unet(self,inp,numCh,nChInc,nlev,'u1',encs[-1]+3)
        out = tf.slice(out,tf.stack([0,ypad[0],xpad[0],0]),tf.stack([-1,fshp[1],fshp[2],-1]))
        out = tf.nn.relu(conv(self,'out',out,[1,1,numCh],0,False)+128.0)
        
        shp = tf.shape(img)
        out1 = gen_nn_ops.avg_pool_grad(tf.stack([fshp[0],fshp[1]*2,fshp[2]*2,1]),out,[1,2,2,1],[1,2,2,1],'VALID')*4.0
        if upsamp:
            out2 = tf.image.resize_bilinear(out,tf.stack([fshp[1]*2,fshp[2]*2]))
            out = tf.where(tf.abs(out1-out2)<1,out2,out1)
        else:
            out = out1
            
        out = tf.slice(out,tf.stack([0,fshp[1]*2-shp[1],fshp[2]*2-shp[2],0]),[-1,-1,-1,-1])
        
        return tf.squeeze(out, axis=-1)
