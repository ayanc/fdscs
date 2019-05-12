#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np

def metrics(output,gt,mask):
    gt,mask = tf.squeeze(gt,-1), tf.squeeze(mask,-1)
    mtot = tf.maximum(1.,tf.reduce_sum(mask))

    loss = tf.pow(tf.maximum(1.0,tf.abs(output - gt)),.125)
    loss = tf.reduce_sum(loss*mask)

    nloss = loss/mtot
    
    dbest = tf.to_float(output)
    L1 = tf.abs(dbest-gt)
    pc = tf.to_float(L1 < 1.5)
    pc = tf.reduce_sum(pc*mask)/mtot
    pc3 = tf.to_float(L1 < 3)
    pc3 = tf.reduce_sum(pc3*mask)/mtot

    L1 = tf.reduce_sum(L1*mask)/mtot


    return loss, nloss, L1, pc, pc3
    
