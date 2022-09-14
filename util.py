from collections import defaultdict


def lines(file, max=None, skip=0, apply_func=str.strip):
    """Iterate over lines in (text) file. Optionally skip first `skip`
    lines, only read the first `max` lines, and apply `apply_func` to
    each line. By default lines are stripped, set `apply_func` to None
    to disable this."""
    from itertools import islice
    if apply_func:
        with open(str(file), encoding="utf8") as f:
            for line in islice(f, skip, max):
                yield apply_func(line)
    else:
        with open(str(file), encoding="utf8") as f:
            for line in islice(f, skip, max):
                yield line

class _Missing:

    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'


_missing = _Missing()


class _cached_property(property):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value:
        class Foo(object):
            @cached_property
            def foo(self):
                # calculate something important here
                return 42
    The class has to have a `__dict__` in order for this property to
    work.
    """
    # source: https://github.com/pallets/werkzeug/blob/master/werkzeug/utils.py

    # implementation detail: A subclass of python's builtin property
    # decorator, we override __get__ to check for a cached value. If one
    # choses to invoke __get__ by hand the property will still work as
    # expected because the lookup logic is replicated in __get__ for
    # manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


def cached_property(func=None, **kwargs):
    # https://stackoverflow.com/questions/7492068/python-class-decorator-arguments
    if func:
        return _cached_property(func)
    else:
        def wrapper(func):
            return _cached_property(func, **kwargs)

        return wrapper


def avg(values):
    n_values = len(values)
    if n_values:
        return sum(values) / n_values
    return 0


def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


class SubclassRegistry:
    '''Mixin that automatically registers all subclasses of the
    given class.
    '''
    classes = dict()
    subclasses = defaultdict(set)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.classes[cls.__name__.lower()] = cls
        for super_cls in cls.__mro__:
            if super_cls == cls:
                continue
            SubclassRegistry.subclasses[super_cls].add(cls)

    @staticmethod
    def get(cls_name):
        return SubclassRegistry.classes[cls_name]

    @classmethod
    def get_subclasses(cls):
        return SubclassRegistry.subclasses.get(cls, {})
