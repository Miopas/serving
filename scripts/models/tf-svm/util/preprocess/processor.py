
class Processor(object):

    def fit(self, x):
        raise NotImplementedError

    def transform(x):
        raise NotImplementedError

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
