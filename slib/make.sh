#!/bin/bash

CUDIR=$(dirname $(dirname $(which nvcc)))
CULIB=${CUDIR}/lib64

rm -rf cuda
ln -s $CUDIR cuda

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 --expt-relaxed-constexpr -c -o slib.cu.o slib.cu.cc -D_GLIBCXX_USE_CXX11_ABI=0 \
    ${TF_CFLAGS[@]} -I . -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o slib.so slib.cc slib.cu.o -D_GLIBCXX_USE_CXX11_ABI=0 \
    ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC -lcudart ${TF_LFLAGS[@]} -L$CULIB

rm cuda
rm slib.cu.o
