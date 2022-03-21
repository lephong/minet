# MINET: a MInimal neural NETwork library for java

This is an educational project for Java users to learn basic
neural networks. The structure of the library, which is mainly 
inspired by [pytorch](pytorch.org), is kept as minimal as possible,
so that 
the learner can easily understand the structure as well as play 
with the library (e.g. adding layers, optimizers). 

Note that: 

- The library is designed without optimization in mind. 
Speed is never a target. 

- It is strongly suggested that the learner starts with a basic 
material about neural networks. [This tutorial](http://ufldl.stanford.edu/tutorial/)
is a great start. 

- Following [pytorch](pytorch.org) and several other deep learning 
libraries, an instance is represented by a *row* vector. 
Therefore, a linear layer is Y = XW + b rather than Y = WX + b.

## Requirement
- [jblas](http://jblas.org/) (tested with v. 1.2.5)

## Tutorials
Please find a number of tutorials in the `tutorial` folder

## Javadoc
A Javadoc can be found [here](https://lephong.github.io/minet/)

