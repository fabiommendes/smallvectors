'''
Mixin classes that can be used to fill up common implementations for other
smallvectors classes.
'''

import abc
from .smallvectorsbase import ABC, Mutable, Immutable, SmallVectorsBase

def mutating(func):
    '''Decorator that marks functions for suitable only for mutable objects'''
     
    func.__mutating__ = True
    return func

    
class Normed(ABC):
    '''Base class for all objects that have a notion of norm'''

    __slots__ = ()

    def __abs__(self):
        return self.norm(None)

    @abc.abstractmethod
    def norm(self, which=None):
        '''Returns the norm of an object.
        
        The `which` attribute controls the method used to compute the norm.'''

        tname = type(self).__name__
        raise TypeError('%s object does not define a norm' % tname)
    
    def norm_sqr(self, which=None):
        '''Returns the squared norm of object.'''

        value = self.norm(which)
        return value * value

    def normalized(self, which=None):
        '''Return a normalized version of object.

        Normalizes according to the given method.'''

        try:
            Z = self.norm(which)
            return self / Z
        except ZeroDivisionError:
            return self
    
    def is_unity(self, tol=1e-6, norm=None):
        '''Return True if the norm equals one within the given tolerance.

        Object must define a normalization method `norm`.'''

        value = self.norm(norm)
        return abs(value - 1) < tol

    @mutating    
    def normalize(self, which=None):
        '''Normalizes object *INPLACE*'''
        
        if isinstance(self, Immutable):
            raise TypeError('cannot normalize immutable object')
        
        self /= self.norm(which)
        

