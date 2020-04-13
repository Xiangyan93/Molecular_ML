class PropertyConfig:
    def __init__(self, property='density'):
        self.property = property
        if property in ['tc', 'dc', 'tt', 'tb', 'pc', 'hfus']:
            self.T, self.P = False, False
        elif property in ['st', 'vis', 'vis_log']:
            self.T, self.P = True, False
        else:
            self.T, self.P = True, True
