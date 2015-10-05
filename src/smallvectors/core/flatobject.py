'''
Some implementations to the .flat attribute of ``smallvectors`` objects. It
can be either a list-like object that prevents some forbiden operations such
as item insertions and deletions or a view to someone that actually owns data.

=====
Usage
=====

A Flat object is very similar to a list

>>> L = Flat([1, 2, 3]); L
flat([1, 2, 3])

A flat view simply exposes data from some other list-like object

>>> view = FlatView(L); view
flat([1, 2, 3])

If we change the data owner, the view updates

>>> flat_setitem(L, 0, 42); L; view
flat([42, 2, 3])
flat([42, 2, 3])

We can even change the view and it is forwarded to the original data owner.
Of course it only works in mutable owners.

>>> L = [1, 2, 3]
>>> view = FlatView(L)
>>> L[0] = 42
>>> print(view)
flat([42, 2, 3])
'''
__author__ = 'Fábio Macêdo Mendes'

import collections

__all__ = ['Flat', 'mFlat', 'FlatView', 'mFlatView', 
           'SimpleFlatView', 'mSimpleFlatView']

def flat_setitem(flat_obj, idx, value):
    '''Used privately to set some item into a Flat object'''

    flat_obj.__flat_setitem__(idx, value)


class FlatAny(collections.Sequence):

    '''A immutable list-like object with private mutation. Must define a
    __flat_setitem__() method to allow private mutation. Flat objects cannot
    be resized.
    '''

    __slots__ = ('_data',)

    def __init__(self, data, copy=True):
        if copy:
            self._data = list(data)
        else:
            self._data = data

    @classmethod
    def from_list_unsafe(cls, L):
        '''Create a new Flat object from a list without copying it. This
        method should only be called if the Flat object can claim ownership
        of the input list.

        There is no guarantee that the flat object will share data with the
        list and caller must not use the input list elsewhere.'''

        new = object.__new__(cls)
        new._data = L
        return new

    def __repr__(self):
        return 'flat([%s])' % (', '.join(map(repr, self)))

    #
    # Flat protocol
    #
    def __flat_setitem__(self, idx, value):
        self._data[idx] = value

    #
    # Sequence interface
    #
    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx_or_slice):
        if isinstance(idx_or_slice, int):
            return self._data[idx_or_slice]
        else:
            return self._data[idx_or_slice]

    def __len__(self):
        return len(self._data)

class Flat(FlatAny):
    '''An immutable flat object'''
    
    
    __slots__ = ()
    
    def __hash__(self):
        return hash(tuple(self))

class mFlat(FlatAny):

    '''An user-mutable flat object. Used to implement the mutable versions of
    ``smallvectors`` objects.'''

    __slots__ = ()

    def __setitem__(self, idx, value):
        self.__flat_setitem__(idx, value)


class SimpleFlatViewAny(FlatAny):
    '''A simple flat facade to some arbitrary sequence-like object.'''

    __slots__ = ('data')

    def __init__(self, data):
        self.data = data
        
    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def __flat_setitem__(self, idx, value):
        self.data[idx] = value


class SimpleFlatView(SimpleFlatViewAny, Flat):
    '''Facade to some arbitrary sequence-like object -- immutable version'''
    
    __slots__ = ()


class mSimpleFlatView(SimpleFlatViewAny, mFlat):
    '''Facade to some arbitrary sequence-like object -- mutable version'''
    
    __slots__ = ()


class FlatViewAny(FlatAny):

    '''A flat facade to some arbitrary sequence-like object.

    It accepts two functions: flat_index and owner_index that maps indexes from
    owner to flat and from flat to owner respectivelly. An explicit size can
    also be given.'''

    __slots__ = ('owner',)

    def __init__(self, owner):
        self.owner = owner

    def __iter__(self):
        try:
            for x in self.owner.__flatiter__():
                yield x
        except AttributeError:
            for i in range(len(self)):
                yield self[i]

    def __len__(self):
        try:
            return self.owner.__flatlen__()
        except AttributeError:
            return len(self.owner)
        
    def __getitem__(self, key):
        owner = self.owner
        try:
            return owner.__flatgetitem__(key)
        except AttributeError:
            if hasattr(owner, '__flatiter__'):
                if isinstance(key, int):
                    N = len(self)
                    if key < 0:
                        key = N - key
                    for i, x in zip(range(N), self):
                        pass
                    if i == key:
                        return x
                    raise IndexError(key)
                
                elif isinstance(key, slice):
                    return [self[i] for i in range(*slice)]
                
                else:
                    raise IndexError('invalid index: %r' % key)
            else:
                return owner[key]

    def __setitem__(self, idx, value):
        try:
            self.owner.__setflatitem__(idx, value)
        except AttributeError: 
            raise TypeError('object must implement __setflatitem__ in order '
                            'to support item assigment' )


class FlatView(FlatViewAny, Flat):

    '''A immutable facade to a sequence-like object'''
    
    __slots__ = ()

class mFlatView(FlatViewAny, mFlat):

    '''A mutable facade to a sequence-like object'''

    __slots__ = ()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
