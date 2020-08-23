class Logger(object):
    def __init__(self,
                 verbose=True
                 ):
        self.verbose = verbose

    def __call__(self, *args):
        if self.verbose:
            print(args)
