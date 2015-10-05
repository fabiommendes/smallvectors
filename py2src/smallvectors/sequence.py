class seq(object):

    '''seq objects describe a range of numbers that can be iterated lazyly.

    Examples
    --------

    In the simplest form, it can create iterables using a natural mathematical
    notation

    >>> numbers = seq(1, 2, Ellipsis, 10)
    >>> list(numbers)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ``seq`` objects have a sequence interface and can be treated just as lazy
    lists

    >>> len(numbers)
    10
    >>> numbers[2]
    3

    '''

    def __init__(self, *args):
        # This function accepts various signatures. Various if's...
        num_args = len(args)
        ellipsis = Ellipsis

        # seq(a, ..., b) --> numbers from a to b, single step. Only works for
        # integer arguments.
        if num_args == 3:
            a, ellipsis, b = args
            self.start = a
            self.end = b
            self.step = 1

        # seq(a, b, ..., c) --> iterate from a to c by (b - a) steps
        elif num_args == 4:
            a, b, ellipsis, c = args
            self.start = a
            self.end = c
            self.step = b - a

        # seq(i, ...) --> all natural numbers starting from i
        # seq(..., i) --> reverse iteration starting from i
        elif num_args == 2:
            self.start, ellipsis = args
            self.step = 1
            self.end = None

            if self.start is Ellipsis:
                self.start = None
                ellipsis, self.start, = args
                self.end = -1

        # seq() --> all natural numbers
        elif num_args == 0:
            self.start = 0
            self.end = None
            self.step = 1

        else:
            raise TypeError('invalid number of arguments: %s')

        if ellipsis is not Ellipsis:
            raise TypeError('expected Ellipsis object')

    def __len__(self):
        try:
            return int((self.end - self.start) // self.step) + 1
        except TypeError:
            if self.end is None:
                raise ValueError('infinite iterator')
            else:
                raise

    def __getitem__(self, idx):
        return self.start + self.step * idx

    def __iter__(self):
        # Finite iterators
        if self.end is not None:
            x, end = self.start, self.end
            step = self.step
            while x <= end:
                yield x
                x += step

        # Infinite iterators
        else:
            x, end = self.start, self.end
            step = self.step
            while x <= end:
                yield x
                x += step


if __name__ == '__main__':
    import doctest
    doctest.testmod()
