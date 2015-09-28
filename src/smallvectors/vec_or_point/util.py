'''
Created on 24/08/2015

@author: chips
'''

from .anyvec import AnyVec, Vec


def asvector(obj):
    '''Return object as a vector'''

    if isinstance(obj, AnyVec):
        return obj
    else:
        return Vec(*obj)
