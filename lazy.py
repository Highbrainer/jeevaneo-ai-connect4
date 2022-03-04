from collections.abc import Mapping
import inspect

### A lazy-loading "dictionary".
### When a key is accessed and its value has been set as as a no-arg function/lambda,
### it is the result of the lambda that is returned as value.
### Lambda are only called once - then result gets cached without expiration.
### Obviously the set value is returned as-if if it is not a zero-arg function.
### Here "no-arg" means that the function can be called without passing a parameter
### (but may have params with default values, *args or **keywords)
### update : if lambda has one required argument, it will be passed the key.
class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        augmented_kw = {key: self.translate_value(value) for key, value in kw.items()}
        self._raw_dict = dict(*args, **augmented_kw)

    def __setitem__(self, key, value):
        self._raw_dict.__setitem__(key, self.translate_value(value))

    def translate_value(self, v):
        if not inspect.isfunction(v):
            # it's not even a function, keep the value as-is
            # print("Not a function !", v)
            return None, v
        nb_required_params = self.get_nb_required_params(v)
        if nb_required_params > 1:
            # it's not a zero-arg function, keep the value as-is
            # print(f'{v.__name__} has {nb_required_params} required params ! (should be zero or one)')
            return None, v
        return v, None

    def get_nb_required_params(self, v):
        argspec = inspect.getfullargspec(v)
        nb_params = len(argspec.args)
        nb_optional_params = 0 if argspec.defaults is None else len(argspec.defaults)
        nb_required_params = nb_params - nb_optional_params
        return nb_required_params

    def __getitem__(self, key):
        func, v = self._raw_dict.__getitem__(key)
        if v is None and not func is None:
            nb_required_params = self.get_nb_required_params(func)
            if nb_required_params == 0:
                v = func()
            elif nb_required_params == 1:
                v = func(key)
            else:
                raise f"Function should have at most one param - it has {nb_required_params} : {key} = {func}"
            self._raw_dict.__setitem__(key, (func, v))
        return v

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)

    def keys(self):
        return self._raw_dict.keys()