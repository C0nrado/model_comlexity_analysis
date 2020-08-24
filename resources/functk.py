from inspect import signature
from functools import partial

def curry(func):
	def curried(*args, func=func, **kwargs):
		func_params = signature(func).parameters
		n_params = len([k for k,val in func_params.items()
					if val.default == val.empty])
		assert n_params >= 1
		if n_params == 1:
			return func(*args, **kwargs)
		else:
			func = partial(func, *args, **kwargs)
			return curry(func)
	return curried

def compose(*functions):
     def composed(functions, arg):
         assert len(functions) >= 1
         if len(functions) == 1:
             return functions[0](arg)
         else:
             func = functions[-1]
             arg = func(arg)
             return composed(functions[:-1], arg)
     return partial(composed, functions)