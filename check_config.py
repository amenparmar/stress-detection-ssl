import importlib
m = importlib.import_module('stress_detection.main')
print('main:', m.__file__)
print('main has WESAD_dataset_path:', hasattr(m, 'WESAD_dataset_path'))
try:
    c = importlib.import_module('stress_detection.utils.config')
    print('config file:', getattr(c, '__file__', None))
    print('config has WESAD_dataset_path:', hasattr(c, 'WESAD_dataset_path'))
    print('config keys:', [k for k in dir(c) if not k.startswith('__')][:50])
except Exception as e:
    print('failed to import config:', e)
