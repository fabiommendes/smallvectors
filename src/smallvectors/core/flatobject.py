import collections
from .parametric import Immutable


__all__ = ['Flat', 'mFlat', 'FlatView']


class Flat(collections.Sequence):

    '''A immutable list-like object that holds a flattened data of a 
    smallvectors object.'''

    __slots__ = ('_data',)

    def __init__(self, data, copy=True):
        if copy:
            self._data = list(data)
        else:
            self._data = data

    def __repr__(self):
        return 'flat([%s])' % (', '.join(map(repr, self)))

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx_or_slice):
        return self._data[idx_or_slice]

    def __len__(self):
        return len(self._data)


class mFlat(Flat):

    '''A mutable Flat object.'''

    __slots__ = ()

    def __setitem__(self, idx, value):
        self._data[idx] = value
        

class FlatView(collections.Sequence):

    '''A flat facade to some arbitrary sequence-like object.

    It accepts two functions: flat_index and owner_index that maps indexes from
    owner to flat and from flat to owner respectivelly. An explicit size can
    also be given.'''

    __slots__ = ('owner',)

    def __init__(self, owner):
        self.owner = owner
        
    def __repr__(self):
        return 'flat([%s])' % (', '.join(map(repr, self)))

    def __iter__(self):
        for x in self.owner.__flatiter__():
            yield x

    def __len__(self):
        return self.owner.size
        
    def __getitem__(self, key):
        owner = self.owner
        try:
            if isinstance(key, int):
                return owner.__flatgetitem__(key)
            else:
                getter = owner.__flatgetitem__
                indices = range(*key.indices(self.owner.size))
                return [getter(i) for i in indices]
        except AttributeError:
            if isinstance(key, int):
                N = owner.size
                if key < 0:
                    key = N - key
                for i, x in zip(range(N), self):
                    if i == key:
                        return x
                else:
                    raise IndexError(key)
            
            elif isinstance(key, slice):
                indices = range(*slice.indices(N))
                return [self[i] for i in indices]
            
            else:
                raise IndexError('invalid index: %r' % key)

    def __setitem__(self, idx, value):
        if isinstance(self.owner, Immutable):
            raise TypeError('cannot change immutable object')
        try:
            setter = self.owner.__flatsetitem__
        except AttributeError: 
            raise TypeError('object must implement __flatsetitem__ in order '
                            'to support item assigment' )
        else:
            setter(idx, value)
