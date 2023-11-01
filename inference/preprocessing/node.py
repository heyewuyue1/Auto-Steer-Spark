class Node:
    def __init__(self) -> None:
        self.lc = None
        self.rc = None
        self.idx = None
        self.operator = None
        self.data = {}


    def __init__(self, idx, operator) -> None:
        self.lc = None
        self.rc = None
        self.idx = idx
        self.operator = operator
        self.data = {}


    def __str__(self) -> str:
        return '('+ str(self.idx) + ') ' + self.operator + '\n' \
            + 'Left Child: ' + str(self.lc) + '\n'\
            + 'Right Child: ' + str(self.rc) + '\n'\
            + '\n'.join([k + ': ' + v for k, v in self.data.items()]) + '\n' \
            + '\n'
