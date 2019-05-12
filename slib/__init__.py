#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import os


sopath=os.path.abspath(os.path.dirname(__file__))+'/slib.so'
mod = tf.load_op_library(sopath)

census = mod.census
hamming = mod.hamming
