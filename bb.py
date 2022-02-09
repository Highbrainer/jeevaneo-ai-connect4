class BB:

    def __init__(self, nb_rows=6, nb_cols=7, initial=0):
        self.clear()
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.bb = initial
        self.size = max(self.nb_rows, self.nb_cols)
        self.mask_col = sum([1 << i for i in range(self.size)])
        self.mask_full = sum([
            1 << (i + i // self.nb_rows)
            for i in range(self.nb_rows * self.nb_cols)
        ])
    
    def __hash__(self):
        return self.bb

    def __eq__(self, other):
        return other == self.bb or (isinstance(other, BB) and self.bb == other.bb)

    def clear(self):
        self.bb = 0

    def count(self):
        cnt = 0
        bb = self.bb
        while bb != 0:
            bb &= bb-1
            cnt += 1
        return cnt


    def get(self, row:int, col:int) -> bool:
        index = row + self.size * col
        ret = ((self.bb >> index) & 1)
        return ret

    def set(self, row:int, col:int) -> int:
        if col >= self.nb_cols:
            raise Exception(
                f"There are only {self.nb_cols} columns, not {col+1} !")
        if row >= self.nb_rows:
            raise Exception(
                f"There are only {self.nb_rows} rows, not {row+1} !")
        index = row + self.size * col
        mask = 1 << int(index)
        self.bb |= mask
        return self.bb

    def getCol(self, col):
        return (self.bb >> (col * self.size)) & self.mask_col

    def getRow(self, row):
        return sum([(self.bb >> (row + self.size * i) & 1) << i
                    for i in range(self.size)])

    def printBB(self):
        print(self.nb_rows, 'x', self.nb_cols, '=>', self.bb)
        for row in range(self.nb_rows - 1, -1, -1):
            print([self.get(row, col) for col in range(self.nb_cols)])

    def hasFour(self):
        # Check \
        temp_bboard = self.bb & (self.bb >> (self.size - 1))
        if (temp_bboard & (temp_bboard >> (2 * (self.size - 1)))):
            return True
        # Check -
        temp_bboard = self.bb & (self.bb >> self.size)
        if (temp_bboard & (temp_bboard >> 2 * self.size)):
            return True
        # Check /
        temp_bboard = self.bb & (self.bb >> (self.size + 1))
        if (temp_bboard & (temp_bboard >> 2 * (self.size + 1))):
            return True
        # Check |
        temp_bboard = self.bb & (self.bb >> 1)
        if (temp_bboard & (temp_bboard >> 2 * 1)):
            return True
        return False

    def isFull(self) -> bool:
        return (self.bb & self.mask_full) == self.mask_full

    def addToColumn(self, col):
        if col >= self.nb_cols:
            raise Exception(
                f"There are only {self.nb_cols} columns, not {col+1} !")
        for row in range(0, self.nb_rows):
            if self.get(row, col) == 0:
                return self.set(row, col)