#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import utils as ut
import model
import pdata as data
import dops

import os.path
import sys

# Params
BSZ = 4
WD = 1e-5
SAVEITER = 1e4
DISPITER = 10
VALITER = 1000
VALREP = 2

saver = ut.ckpter('wts/model*.npz')
LR = 1e-5
MAXITER = 5e3
if saver.iter >= MAXITER:
    MAXITER=350e3
    LR = 1e-4
    
#### Build Graph

# Build phase2 
d = data.dataset(BSZ)
net = model.Net()
output = net.predict(d.limgs, d.cv, d.lrl)
tloss, loss, l1, pc, pc3 = dops.metrics(output,d.disp,d.mask)

vals = [loss,pc,l1,pc3]
tnms = ['loss.t','pc.t','L1.t','pc3.t']
vnms = ['loss.v','pc.v','L1.v','pc3.v']

opt = tf.train.AdamOptimizer(LR)
tstep = opt.minimize(tloss+WD*net.wd,var_list=list(net.weights.values()))

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
sess.run(tf.global_variables_initializer())

# Load Data File Names
tlist = [f.rstrip('\n') for f in open('data/ftrain.txt').readlines()]
vlist = [f.rstrip('\n') for f in open('data/fval.txt').readlines()]

ESIZE=len(tlist)//BSZ
VESIZE=len(vlist)//BSZ

# Setup save/restore
origiter = saver.iter
rs = np.random.RandomState(0)
if origiter > 0:
    ut.loadNet(saver.latest,net,sess)
    if os.path.isfile('wts/popt.npz'):
        ut.loadAdam('wts/popt.npz',opt,net.weights,sess)
    
    for k in range( (origiter+ESIZE-1) // ESIZE):
        idx = rs.permutation(len(tlist))
    ut.mprint("Restored to iteration %d" % origiter)    

    
# Main Training Loop    
niter = origiter    
touts = 0.
while niter < MAXITER+1:
    
    if niter % VALITER == 0:
        vouts = 0.
        for j in range(VALREP):
            off = j % (len(vlist)%BSZ + 1)
            for b in range(VESIZE):
                blst = vlist[(b*BSZ+off):((b+1)*BSZ+off)]
                outs = sess.run(vals,feed_dict=d.fdict(blst))
                vouts = vouts + np.float32(outs)
                
        vouts = vouts / np.float32(VESIZE*VALREP)
        ut.vprint(niter,vnms,vouts)

    if niter == MAXITER:
        break

    if niter % ESIZE == 0:
        idx = rs.permutation(len(tlist))

    blst = [tlist[idx[(niter%ESIZE)*BSZ+b]] for b in range(BSZ)]
    outs,_ = sess.run([vals,tstep],feed_dict=d.fdict(blst))
    niter = niter+1
    touts = touts+np.float32(outs)

    if niter % SAVEITER == 0:
        ut.saveNet('wts/model_%d.npz'%niter,net,sess)
        saver.clean(every=SAVEITER,last=1)
        ut.mprint('Saved Model')
        
    if niter % DISPITER == 0:
        touts = touts/np.float32(DISPITER)
        ut.vprint(niter,['lr']+tnms,[LR]+list(touts))
        touts = 0.
        if ut.stop:
            break
        
if niter > saver.iter:
    ut.saveNet('wts/model_%d.npz'%niter,net,sess)
    saver.clean(every=SAVEITER,last=1)
    ut.mprint('Saved Model')
    
if niter > origiter:
    ut.saveAdam('wts/popt.npz',opt,net.weights,sess)
    ut.mprint("Saved Optimizer.")
