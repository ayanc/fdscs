#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import utils as ut
import model
import data

import os.path
import sys

# Params
BSZ = 4
MAXITER=500

d = data.dataset(BSZ)
net = model.Net()
_ = net.predict(d.limgs, d.cv, d.lrl)

blyrs = list(net.bnvals.keys())
blyrs = [nm[:-2] for nm in blyrs if nm[-2:] == '_m']

bnms = [[] for k in range(len(blyrs))]
bnms_ = [net.bnvals[blyrs[k] + '_m'] for k in range(len(blyrs))]
bnvs = [[] for k in range(len(blyrs))]
bnvs_ = [net.bnvals[blyrs[k] + '_v'] for k in range(len(blyrs))]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load Data File Names
tlist = [f.rstrip('\n') for f in open('data/train.txt').readlines()]
ESIZE=len(tlist)//BSZ

rs = np.random.RandomState(0)
ut.loadNet(sys.argv[1],net,sess)
ut.mprint("Loaded weights from " + sys.argv[1])    

# Loop through a bunch of batches
ut.mprint("Running forward")
niter = 0
while niter < MAXITER:
    ut.mprint("Running iteration %d" % niter)
    if niter % ESIZE == 0:
        idx = rs.permutation(len(tlist))

    blst = [tlist[idx[(niter%ESIZE)*BSZ+b]] for b in range(BSZ)]
    ms, vs = sess.run([bnms_,bnvs_],feed_dict=d.fdict(blst))
    for k in range(len(blyrs)):
        bnms[k].append(ms[k])
        bnvs[k].append(vs[k])
    
    niter = niter+1

ut.mprint("Updating Weights")
wts = np.load(sys.argv[1])
wts = {k: wts[k] for k in wts.keys()}

for k in range(len(blyrs)):
    mk,vk = np.float32(np.stack(bnms[k],0)), np.float32(np.stack(bnvs[k],0))

    vk = np.mean(vk,0) + np.var(mk,0)
    mk = np.mean(mk,0)
    fac = 1.0/np.sqrt(model.bneps+vk)
    
    wts[blyrs[k]+"_w"] *= fac
    wts[blyrs[k]+"_b"] -= mk*fac/2.0

np.savez(sys.argv[2],**wts)    
