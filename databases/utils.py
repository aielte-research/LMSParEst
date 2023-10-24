from functools import reduce
import operator


def product(iterable):
    return reduce(operator.mul, iterable, 1)


def cond(predicate, consequence, alternative):
    if predicate:
        return consequence
    else:
        return alternative


def cond_exception(predicate, consequence, exception):
    if predicate:
        return consequence
    else:
        raise Exception(exception)


def in_interval(interval, value):
    return cond((interval[0] <= value and interval[1] >= value), True, False)


def sub_interval(main, sub):
    return cond((main[0] <= sub[0] and main[1] >= sub[1]), True, False)

def excep_raiser(statement,exception):
    if not statement:
        return Exception(exception)
        
def my_arraycacher(*args, **kwargs):
# lru caching decorator for unhashable numpy array object, it is a work in progress
    def decorator(function):
        @wraps(function)
        def wrapper(s, unhashable_array, *args, **kwargs):
            hashable_array = tuple(map(tuple, unhashable_array))
            return cached_wrapper(s, hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(s, hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(s, array, *args, **kwargs)
        
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator

 
    
