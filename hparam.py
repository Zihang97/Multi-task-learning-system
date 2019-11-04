import yaml

def load_hparam(filename):
    f = open(filename, 'r')
    docc = dict()
    docs = yaml.load_all(f)
    for doc in docs:
        docc.update(doc)
    return docc

class Dotdict(dict):

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():

            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='config/config.yaml'):
        super(Dotdict, self).__init__()

        hp_dict = load_hparam(file)

        hp_dotdict = Dotdict(hp_dict)

        for k, v in hp_dotdict.items():
            setattr(self, k, v)
    #
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    
hparam = Hparam()
