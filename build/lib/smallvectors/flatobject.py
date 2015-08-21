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

>>> L = MutableFlat([1, 2, 3])
>>> view = FlatView(L)
>>> flat_setitem(view, 0, 42)  # notice we are changing view, not L!
>>> print(L)
flat([42, 2, 3])
'''
__author__ = 'Fábio Macêdo Mendes'

import collections


def flat_setitem(flat_obj, idx, value):
    '''Used privately to set some item into a Flat object'''

    flat_obj.__flat_setitem__(idx, value)


class Flat(collections.Sequence):

    '''A immutable list-like object with private mutation. Must define a
    __flat_setitem__() method to allow private mutation. Flat objects cannot
    be resized.
    '''

    __slots__ = '_data'

    def __init__(self, data):
        self._data = list(data)

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


class MutableFlat(Flat):

    '''An user-mutable flat object. Used to implement the mutable versions of
    ``smallvectors`` objects.'''

    __slots__ = []

    def __setitem__(self, idx, value):
        self.__flat_setitem__(idx, value)


class FlatView(Flat):

    '''A flat facade to some arbitrary sequence-like object.

    It accepts two functions: flat_index and owner_index that maps indexes from
    owner to flat and from flat to owner respectivelly. An explicit size can
    also be given.'''

    __slots__ = ['_owner', '_owner_index', '_flat_index', '_size']

    def __init__(self, owner, size=None, flat_index=None, owner_index=None):
        self._owner = owner
        self._flat_index = flat_index or (lambda x: x)
        self._owner_index = owner_index or (lambda x: x)
        self._size = len(owner) if size is None else size

    #
    # Flat protocol
    #
    def __flat_setitem__(self, idx, value):
        self._owner[self._owner_index(idx)] = value

    #
    # Sequence interface
    #
    def __iter__(self):
        owner = self._owner
        owner_index = self._owner_index

        for idx in range(self._size):
            yield owner[owner_index(idx)]

    def __getitem__(self, idx_or_slice):
        owner = self._owner
        owner_index = self._owner_index

        if isinstance(idx_or_slice, int):
            idx = idx_or_slice
            return owner[owner_index(idx)]
        else:
            slice = idx_or_slice
            indices = slice.indices(self._size)
            return [owner[owner_index(idx)] for idx in indices]

    def __len__(self):
        return self._size


class FlatMutableView(FlatView, MutableFlat):

    '''A mutable facade to a sequence-like object'''

    __slots__ = []

if __name__ == '__main__':
    import doctest
    doctest.testmod()
