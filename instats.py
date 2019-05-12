#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import utils as ut
import data

import os.path
import sys

# Params
BSZ = 4
MAXITER=10

d = data.dataset(BSZ)
cv = tf.reshape(d.cv,[-1,3])
mu,v = tf.nn.moments(cv,[0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load Data File Names
tlist = [f.rstrip('\n') for f in open('data/train.txt').readlines()]
ESIZE=len(tlist)//BSZ

rs = np.random.RandomState(0)



mus, vs = [], []
# Loop through a bunch of batches
ut.mprint("Running forward")
niter = 0
while niter < MAXITER:
    ut.mprint("Running iteration %d" % niter)
    if niter % ESIZE == 0:
        idx = rs.permutation(len(tlist))

    blst = [tlist[idx[(niter%ESIZE)*BSZ+b]] for b in range(BSZ)]
    mv, vv = sess.run([mu,v],feed_dict=d.fdict(blst))
    mus.append(mv)
    vs.append(vv)
    niter = niter+1

mus,vs = np.float32(np.stack(mus,0)), np.float32(np.stack(vs,0))

vs = np.mean(vs,0) + np.var(mus,0)
vs = 1./np.sqrt(vs)
mus = np.mean(mus,0)


print(mus)
print(vs)
