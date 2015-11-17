'''
=======
Vectors
=======


'''
import operator
from generic import promote, set_promotion, set_conversion, get_conversion, convert
from generic.errors import InexactError
from generic.op import add, sub
from .tools import dtype as _dtype
from .core import FlatView
from .core import Immutable, Mutable, Normed, AddElementWise, MulScalar
from .core import SmallVectorsBase


__all__ = ['VecAny', 'mVec', 'Vec', 'asvector', 'asmvector', 'asavector']
DIMENSION_BASES = {}


class VecAny(SmallVectorsBase, Normed, AddElementWise, MulScalar):

    '''Common implementations for Vec and Point types'''

    __slots__ = ()
    __parameters__ = (int, type)
    __abstract__ = True

    @classmethod
    def __preparebases__(cls, params):
        try:
            bases = (DIMENSION_BASES[params[0]], cls)
        except KeyError:
            bases = (cls,) 
        return bases
    
    @classmethod
    def __preparenamespace__(cls, params):
        ns = SmallVectorsBase.__preparenamespace__(params)
        if isinstance(params[0], int) and params[0] > 4:
            ns.update(__slots__='flat')
        return ns
    
    @staticmethod
    def __finalizetype__(cls):
        SmallVectorsBase.__finalizetype__(cls)
        
        # Make all x, y, z, etc property accessors. Mutable objects have
        # read/write accessors
        def getter(i):
            return lambda self: self[i]
        def setter(i):
            return lambda self, value: self.__setitem__(i, value)
        
        for i in range(cls.size or 0):
            prop = property(getter(i))
            if issubclass(cls, Mutable):
                prop = prop.setter(setter(i))
            
            attr = 'x%i' % i
            if not hasattr(cls, attr):
                setattr(cls, attr, prop)
            if i < 4:
                attr = 'xyzw'[i]
                if not hasattr(cls, attr):
                    setattr(cls, attr, prop)
        
    @classmethod
    def __abstract_new__(cls, *args, dtype=None):  # @NoSelf
        '''
        This function is called when user tries to instatiate an abstract
        type. It just finds the proper concrete type and instantiate it.
        '''
        if dtype is None:
            dtype = _dtype(args)
        return cls[len(args), dtype](*args)

    #
    # Representation and conversions
    #
    def as_vector(self):
        '''Returns a copy of object as a vector'''

        return self.__vector__(*self)

    def as_direction(self):
        '''Returns a normalized copy of object as a direction'''

        return self.__direction__(*self)

    def as_point(self):
        '''Returns a copy of object as a point'''

        return self.__point__(*self)

    def copy(self, x=None, y=None, z=None, w=None, **kwds):
        '''Return a copy possibly overriding some components'''

        data = list(self)
        if x is not None:
            data[0] = x
        if y is not None:
            data[1] = y
        if z is not None:
            data[2] = z
        if w is not None:
            data[3] = w

        # Keywords of the form e1=foo, e2=bar, etc
        for k, v in kwds.items():
            if not k.startswith('x'):
                raise TypeError('invalid argument %r' % k)
            data[int(k[1:]) + 1] = v

        return self.fromflat(data)

    #
    # Geometric properties
    #
    def almost_equal(self, other, tol=1e-3):
        '''Return True if two smallvectors are almost equal to each other'''

        return (self - other).norm_sqr() < tol * tol

    def distance(self, other):
        '''Computes the distance between two objects'''

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise IndexError('invalid dimensions: %s and %s' % (N, M))

        deltas = (x - y for (x, y) in zip(self, other))
        return self._sqrt(sum(x * x for x in deltas))

    def lerp(self, other, weight):
        '''Linear interpolation between two objects.

        The weight attribute defines how close the result will be from the
        argument. A weight of 0.5 corresponds to the middle point between
        the two objects.'''

        if not 0 <= weight <= 1:
            raise ValueError('weight must be between 0 and 1')

        return (other - self) * weight + self

    def middle(self, other):
        '''The midpoint to `other`. The same as ``obj.lerp(other, 0.5)``'''

        return (self + other) / 2
    
    def displaced(self, *args):
        '''Return a copy of object displaced by the given ammount.'''
        
        if len(args) == 1:
            args = args[0]
        return self + args
        
    def moved(self, *args):
        '''An alias to obj.displaced()'''
        
        return self.displaced(*args)
    
    def angle(self, other):
        '''Angle between two vectors'''

        try:
            Z = other.norm()
        except AttributeError:
            other = self.vector_type(other)
            Z = other.norm()

        cos_t = self.dot(other) / (self.norm() * Z)
        return self._acos(cos_t)

    def reflected(self, direction):
        '''Reflect vector around the given direction'''

        return self - 2 * (self - self.project(direction))

    def projection(self, direction):
        '''Returns the projection vector in the given direction'''

        direction = self.to_direction(direction)
        return self.dot(direction) * direction

    def clampped(self, min_length, max_length=None):
        '''Returns a new vector in which min_length <= abs(out) <= max_length.'''

        if max_length is None:
            ratio = min_length / self.norm()
            return self * ratio

        norm = new_norm = self.norm()
        if norm > max_length:
            new_norm = max_length
        elif norm < min_length:
            new_norm = min_length

        if new_norm != norm:
            return self.normalized() * new_norm
        else:
            return self.copy()

    def dot(self, other):
        '''Dot product between two objects'''

        if len(self) != len(other):
            N, M = len(self), len(other)
            raise ValueError('dimension mismatch: %s and %s' % (N, M))
        return sum(x * y for (x, y) in zip(self, other))

    def norm(self, which=None):
        '''Returns the norm of a vector'''

        if which is None or which == 'euclidean':
            return self._sqrt(self.norm_sqr())
        elif which == 'minkowski':
            return max(map(abs, self))
        elif which == 'max':
            return max(map(abs, self))
        elif which == 'minkowski':
            return sum(map(abs, self))
        else:
            super().norm(which)

    def norm_sqr(self, which=None):
        '''Returns the squared norm of a vector'''

        if which is None:
            return sum(x * x for x in self)
        else:
            super().norm_sqr(which)

    def rotated(self, rotation):
        '''Rotate vector by the given rotation object.
        
        The rotation can be specified in several ways which are dimension 
        dependent. All vectors can be rotated by rotation matrices in any 
        dimension. 
    
        In 2D, rotation can be a simple number that specifies the 
        angle of rotation.   
        '''
        
        if isinstance(rotation, self._rotmatrix):
            return rotation * self
        
        tname = type(self).__name__
        msg = 'invalid rotation object for %s: %r' % (tname, rotation)
        raise TypeError(msg)
    
    
#
# Special overrides for vectors of specific dimensions
#
class VecND:
    '''
    Base class with overrides for any dimension
    '''
    
    @property
    def flat(self):
        return FlatView(self)
    
    def __flatiter__(self):
        return iter(self)

    def __flatlen__(self):
        return len(self)
    
    def __flatgetitem__(self, idx):
        return self[idx]
    
    def __flatsetitem__(self, idx, value):
        self[idx] = value


class Vec0D(VecND):
    '''
    0D Vectors (probably even less necessary than Vec1D :P)
    '''
    
    __slots__ = ()
    
    def __init__(self):
        pass
    
    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())
    
    def __getitem__(self, idx):
        raise IndexError(idx)


class Vec1D(VecND):
    '''
    1D Vectors (is this necessary?)
    '''
    __slots__ = ('_x')
    
    def __init__(self, x):
        self._x = convert(x, self.dtype)
        
    def __len__(self):
        return 1

    def __iter__(self):
        yield self._x
        
    def __getitem__(self, idx):
        if idx == 0:
            return self._x
        return super().__getitem__(idx)
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        _assure_mutable_set_coord(self)
        self._x = convert(value, self.dtype)
        
    x0 = x


class Vec2D(VecND):
    '''Vector functions that only works in 2D.
    
    These functions are inserted to all Vec[2, ...] classes upon class 
    creation'''
    
    __slots__ = ('_x', '_y')
    
    def __init__(self, x, y):
        self._x = convert(x, self.dtype)
        self._y = convert(y, self.dtype)
        
    def __len__(self):
        return 2

    def __iter__(self):
        yield self._x
        yield self._y
        
    def __getitem__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        return super().__getitem__(idx)

    def __addsame__(self, other):
        return self._fromcoords_unsafe(self._x + other._x, self._y + other._y)

    def __subsame__(self, other):
        return self._fromcoords_unsafe(self._x - other._x, self._y - other._y)

    def __mul__(self, other):
        x = self._x * other
        y = self._y * other
        if isinstance(x, self.dtype):
            return self._fromcoords_unsafe(x, y)
        elif isinstance(other, self._number):
            return self.__origin__(x, y)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self * (1.0 / other)

    #
    # Constructors
    #
    @classmethod
    def fromflat(cls, data, copy=True):
        x, y = data
        return cls._fromcoords_unsafe(x, y)
    
    @classmethod
    def frompolar(cls, radius, theta=0):
        '''Create vector from polar coordinates'''
        return cls(radius * cls._cos(theta), radius * cls._sin(theta))

    @classmethod
    def _fromcoords_unsafe(cls, x, y):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        return new
    
    #
    # Properties
    #
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        _assure_mutable_set_coord(self)
        self._y = convert(value, self.dtype)
    
    x = x0 = Vec1D.x
    x1 = y

    #
    # 2D specific API
    #
    def rotated_axis(self, axis, theta):
        '''Rotate vector around given axis by the angle theta'''

        dx, dy = self - axis
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        return self._fromcoords_unsafe(
            dx * cos_t - dy * sin_t + axis[0],
            dx * sin_t + dy * cos_t + axis[1])

    def rotated(self, theta):
        '''Rotate vector by an angle theta around origin'''

        if isinstance(theta, self._rotmatrix):
            return theta * self
        
        cls = type(self)
        x, y = self
        cos_t, sin_t = self._cos(theta), self._sin(theta)
        
        # TODO: decent implementation of this!
        try:
            return cls(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
        except InexactError:
            if isinstance(self, Mutable):
                return mVec(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
            else:
                return Vec(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
    
    def cross(self, other):
        '''The z component of the cross product between two bidimensional
        smallvectors'''

        x, y = other
        return self.x * y - self.y * x
    
    def polar(self):
        '''Return a tuple with the (radius, theta) polar coordinates '''

        return (self.norm(), self._atan2(self.y, self.x))

    def perp(self, ccw=True):
        '''Return the counterclockwise perpendicular vector.

        If ccw is False, do the rotation in the clockwise direction.
        '''

        if ccw:
            return self._fromcoords_unsafe(-self.y, self.x)
        else:
            return self._fromcoords_unsafe(self.y, -self.x)

    #
    # Performance overrides
    #
    def distance(self, other):
        return self._sqrt((other.x - self._x) ** 2 + (other.y - self._y) ** 2)

    def convert(self, dtype):
        _float = float

        if dtype is self.dtype:
            return self
        elif dtype is _float:
            x = _float(self._x)
            y = _float(self._y)

            try:
                return self.__float_root(x, y)
            except AttributeError:
                self_t = type(self)
                cls = self_t.__float_root = self.__origin__[2, float]
                return cls(x, y)
        else:
            conv = get_conversion(self.dtype, dtype)
            x = conv(self._x)
            y = conv(self._y)
            return self.__origin__[2, dtype](x, y)

    def angle(self, other):
        '''Computes the angle between two smallvectors'''

        cos_t = self.dot(other)
        sin_t = self.cross(other)
        return self._atan2(sin_t, cos_t)

    def is_null(self):
        '''Checks if vector has only null components'''

        if self._x == 0.0 and self._y == 0.0:
            return True
        else:
            return False

    def is_unity(self, tol=1e-6):
        '''Return True if the norm equals one within the given tolerance'''

        return abs(self._x * self._x + self._y * self._y - 1) < 2 * tol

    def norm(self, which=None):
        '''Returns the norm of a vector'''

        if which is None:
            return self._sqrt(self._x ** 2 + self._y ** 2)
        else:
            return Vec.norm(self, which)

    def norm_sqr(self, which=None):
        '''Returns the squared norm of a vector'''

        if which is None:
            return self._x ** 2 + self._y ** 2
        else:
            return Vec.norm(self, which)


class Vec3D(VecND):
    '''Vector functions that only works in 3D.
    
    These functions are inserted to all Vec[3, ...] classes upon class 
    creation'''
    
    __slots__ = ('_x', '_y', '_z')
    
    def __init__(self, x, y, z):
        dtype = self.dtype
        self._x = convert(x, dtype)
        self._y = convert(y, dtype)
        self._z = convert(z, dtype)

    def __len__(self):
        return 3

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z
        
    def __getitem__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        elif idx == 2:
            return self._z
        raise IndexError(idx)

    @classmethod
    def fromflat(cls, data, copy=True):
        x, y, z = data
        return cls._fromcoords_unsafe(x, y, z)
    
    @classmethod
    def fromspheric(cls, radius, phi=0, theta=0):
        '''Create vector from spherical coordinates'''
        
        r = radius * cls._sin(phi)
        x = r * cls._cos(theta) 
        y = r * cls._sin(theta)
        z = r * cls._cos(phi)
        return cls(x, y, z)

    @classmethod
    def fromcylindric(cls, radius, theta=0, z=0):
        '''Create vector from cylindric coordinates'''
        
        x = radius * cls._cos(theta)
        y = radius * cls._sin(theta)
        return cls(x, y, z)

    @classmethod
    def _fromcoords_unsafe(cls, x, y, z):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        new._z = z
        return new
    
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        _assure_mutable_set_coord(value)
        self._z = value

    x = x0 = Vec2D.x
    y = x1 = Vec2D.y
    x3 = z
    
    def cross(self, other):
        '''The cross product between two tridimensional smallvectors'''

        x, y, z = self
        a, b, c = other
        return Vec(y * c - z * b, z * a - x * c, x * b - y * a)

class Vec4D(VecND):
    '''Vector functions that only works in 4D.
    
    These functions are inserted to all Vec[4, ...] classes upon class 
    creation'''

    def __init__(self, x, y, z, w):
        dtype = self.dtype
        self._x = convert(x, dtype)
        self._y = convert(y, dtype)
        self._z = convert(z, dtype)
        self._w = convert(w, dtype)

    def __len__(self):
        return 4

    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z
        yield self._w

    def __getitem__(self, idx):
        if idx == 0:
            return self._x
        elif idx == 1:
            return self._y
        elif idx == 2:
            return self._z
        elif idx == 3:
            return self._w
        raise IndexError(idx)

    @classmethod
    def fromflat(cls, data, copy=True):
        x, y, z, w = data
        return cls._fromcoords_unsafe(x, y, z, w)
    
    @classmethod
    def fromspheric(cls, radius, phi=0, theta=0):
        '''Create vector from spherical coordinates'''
        
        r = radius * cls._sin(phi)
        x = r * cls._cos(theta) 
        y = r * cls._sin(theta)
        z = r * cls._cos(phi)
        return cls(x, y, z)

    @classmethod
    def fromcylindric(cls, radius, theta=0, z=0):
        '''Create vector from cylindric coordinates'''
        
        x = radius * cls._cos(theta)
        y = radius * cls._sin(theta)
        return cls(x, y, z)

    @classmethod
    def _fromcoords_unsafe(cls, x, y, z):
        new = object.__new__(cls)
        new._x = x
        new._y = y
        new._z = z
        return new

    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, value):
        _assure_mutable_set_coord(value)
        self._w = value
    
    x = x0 = Vec3D.x
    y = x1 = Vec3D.y
    z = x2 = Vec3D.z
    x3 = w

#
# User-facing types
#
class Vec(VecAny, Immutable):

    '''Base class for all immutable vector types. Each dimension and type have
    its own related class. '''

    __slots__ = ()


class mVec(VecAny, Mutable):

    '''A mutable vector'''

    __slots__ = ()

    def rotate(self, rotation):
        '''Similar to obj.rotate(...), but make changes *INPLACE*'''
        
        value = self.rotated(rotation)
        self[:] = value
        
    def move(self, *args):
        '''Alias to obj.displace(...)'''
        
        self.displace(*args)
    
    def displace(self, *args):
        '''Similar to obj.move(...), but make changes *INPLACE*'''
        
        if len(args) == 1:
            args = args[0]
        self += args


#
# Maps dimensions to additional bases
#
DIMENSION_BASES = {0: Vec0D, 1: Vec1D, 2: Vec2D, 3: Vec3D, 4: Vec4D}


#
# Promotions and function overloads
#
@set_promotion(Vec, Vec)
def promote_vectors(u, v):
    '''Promote two Vec types to the same type'''

    u_type = type(u)
    v_type = type(v)
    if u_type is v_type:
        return (u, v)

    # Test shapes
    if u_type.shape != v_type.shape:
        raise TypeError('vectors have different shapes')

    # Fasttrack common cases
    u_dtype = u.dtype
    v_dtype = v.dtype
    if u_dtype is float and v_dtype is int:
        return u, v.convert(float)
    elif u_dtype is int and v_dtype is float:
        return u.convert(float), v

    zipped = [promote(x, y) for (x, y) in zip(u, v)]
    u = Vec(*[x for (x, y) in zipped])
    v = Vec(*[y for (x, y) in zipped])
    return u, v

# Conversions and promotions between vec types and tuples/lists
set_conversion(Vec, tuple, tuple)
set_conversion(Vec, list, list)
for T in [Vec, mVec]:
    set_conversion(tuple, T, T)
    set_conversion(list, T, T)


@set_promotion(Vec, tuple, symmetric=True, restype=Vec)
@set_promotion(Vec, list, symmetric=True, restype=Vec)
def promote(u, v):
    return u, u.__origin__.from_seq(v)


def asvector_overload(op, tt):
    real_op = getattr(operator, op.__name__)

    @op.overload((Vec, tt))
    def overload(u, v):
        return real_op(u, Vec(*v))

    @op.overload((tt, Vec))
    def overload(u, v):  # @DuplicatedSignature
        return real_op(Vec(*u), v)

for op in [add, sub]:
    for tt in [tuple, list]:
        asvector_overload(op, tt)

#
# Helper functions
#
def _assure_mutable_set_coord(obj):
    if isinstance(obj, Immutable):
        raise AttributeError('cannot set coordinate of immutable object')


#
# Vector conversions
#
def asvector(obj):
    '''Return object as an immutable vector'''

    if isinstance(obj, Vec):
        return obj
    else:
        return Vec(*obj)

def asmvector(obj):
    '''Return object as a mutable vector'''

    if isinstance(obj, mVec):
        return obj
    else:
        return mVec(*obj)

def asavector(obj):
    '''Return object as a vector.
    
    Non-Vec objects are converted to immutable vectors.'''

    if isinstance(obj, VecAny):
        return obj
    else:
        return Vec(*obj)
