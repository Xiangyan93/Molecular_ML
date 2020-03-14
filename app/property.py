class PropertyConfig:
    def __init__(self, property='density'):
        self.property = property
        if property in ['tc', 'dc', 'tt', 'tb']:
            self.T, self.P = False, False
        elif property in ['st']:
            self.T, self.P = True, False
        else:
            self.T, self.P = True, True
