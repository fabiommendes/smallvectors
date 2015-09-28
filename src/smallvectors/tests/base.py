# -*- coding: utf8 -*-
import operator
from warnings import warn
try:
    import unittest2 as unittest
except ImportError:
    import unittest

opmap = {
    '+': operator.add, 'add': operator.add,
    '-': operator.sub, 'sub': operator.sub,
    '*': operator.mul, 'mul': operator.mul,
    '/': operator.truediv, 'div': operator.truediv,
}
op_inv_map = {v: k for (k, v) in opmap.items()}


class ArithmeticUnittest(unittest.TestCase):
    '''
    Base class for testing mathematical objects that support arithmetical
    operations

    >>> class TestInt(ArithmeticUnittest):                     # doctest: +SKIP
    ...    obj_type = int
    ...
    ...    def names(self):
    ...        return dict(
    ...            x=1, y=2, add_xy=3, sub_xy=-1, sub_yx=1, mul_xy=2,
    ...        )
    '''

    obj_type = None
    commutes = ['add', 'mul']
    str_equality = False
    equal_alternatives = True

    def __init__(self, *args, **kwds):
        self_t = type(self)
        if (self.obj_type is None and not (
                self_t is ArithmeticUnittest or 'Abstract' in self_t.__name__)):
            tname = type(self).__name__
            raise RuntimeError('object type must be specified in the '
                               '"obj_type" class attribute of %s' % tname)
        super(ArithmeticUnittest, self).__init__(*args, **kwds)

    def names(self):
        self_t = type(self)
        if self_t is not ArithmeticUnittest:
            warn('you should override the names() method that returns a '
                 'dictionary of names for test values.')
        return {}

    def setUp(self):
        self._names = self.names()
        for k, v in self._names.items():
            setattr(self, k, v)

    def bin_examples(self, name,
                     commutes=False, alternatives=False, scalar=False):
        '''Retorna um iterador sobre os exemplos (a, b, c) onde
        bin(a, b) == c, para um operador binário fornecido'''

        # Extrai todos os resultados registrados
        prefix = '%s_' % name
        names = self.names()
        res = {k[4:]: v for (k, v) in names.items() if k.startswith(prefix)}

        # Itera sobre os resultados para selecionar os operandos
        obj_tt = self.obj_type
        for k, res in res.items():
            a, b = [names[c] for c in k]

            # Caso scalar esteja ligado, só utiliza os resultados em que um
            # dos membros do par não seja de obj_tt
            if scalar:
                if isinstance(a, obj_tt) and isinstance(b, obj_tt):
                    continue
            else:
                if ((not alternatives) and ((not isinstance(a, obj_tt)) or
                                            (not isinstance(b, obj_tt)))):
                    continue

            if alternatives:
                name_a, name_b = k

                # Retorna todas permutações com a
                a_alts = [v for (k, v) in names.items()
                          if k.startswith(name_a + '_')]
                for a_alt in a_alts:
                    if isinstance(b, obj_tt) or isinstance(a_alt, obj_tt):
                        if commutes:
                            yield (b, a_alt, res)
                        yield (a_alt, b, res)

                # Retorna todas permutações com b
                b_alts = [v for (k, v) in names.items()
                          if k.startswith(name_b + '_')]
                for b_alt in b_alts:
                    if isinstance(a, obj_tt) or isinstance(b_alt, obj_tt):
                        if commutes:
                            yield (b_alt, a, res)
                        yield (a, b_alt, res)

            else:
                # Retorna a comutação
                if commutes:
                    yield (b, a, res)
                yield (a, b, res)

    def bin_assert(self, op, a, b, res):
        value = opmap[op](a, b)
        msg = '%s(%s, %s) => %s, expect %s' % (op, a, b, value, res)
        assert self.equals(value, res), msg

    def bin_worker(self, op, **kwds):
        commutes = op in self.commutes
        for a, b, res in self.bin_examples(op, commutes=commutes):
            self.bin_assert(op, a, b, res)

    def equals(self, a, b):
        if a == b:
            return True
        elif hasattr(a, 'almost_equal'):
            if a.almost_equal(b):
                return True
        elif hasattr(b, 'almost_equal'):
            if b.almost_equal(a):
                return True
        elif self.str_equality and str(a) == str(b):
            return True
        else:
            return False

    # Operações binárias de tipos iguais ######################################
    def test_add(self):
        self.bin_worker('add')

    def test_sub(self):
        self.bin_worker('sub')

    def test_mul(self):
        self.bin_worker('mul')

    def test_div(self):
        self.bin_worker('div')

    # Operações binárias com tipos escalares ##################################
    def test_add_scalar(self):
        self.bin_worker('add', scalar=True)

    def test_sub_scalar(self):
        self.bin_worker('sub', scalar=True)

    def test_mul_scalar(self):
        self.bin_worker('mul', scalar=True)

    def test_div_scalar(self):
        self.bin_worker('div', scalar=True)

    # Operações binárias de tipos alternativos ################################
    def test_add_alts(self):
        self.bin_worker('add', alternatives=True)

    def test_sub_alts(self):
        self.bin_worker('add', alternatives=True)

    def test_mul_alts(self):
        self.bin_worker('add', alternatives=True)

    def test_div_alts(self):
        self.bin_worker('add', alternatives=True)

    # Testa igualdade com alternativas ########################################
    def test_equal_alternatives(self):
        if self.equal_alternatives:
            names = self.names()
            objs = [(k, v) for (k, v) in names.items() if len(k) == 1 and
                    isinstance(v, self.obj_type)]
            for name, obj in objs:
                prefix = name + '_'
                alts = [v for (k, v) in names.items() if k.startswith(prefix)]
                for alt in alts:
                    msg = '%s != %s' % (obj, alt)
                    assert obj == alt, msg


if __name__ == '__main__':
    class TestInt(ArithmeticUnittest):
        obj_type = int

        def names(self):
            return dict(
                x=1, y=2, add_xy=3, sub_xy=-1, sub_yx=1, mul_xy=2,
            )
    unittest.main()
