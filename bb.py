# decorator to call static constructor if any
def static_init(cls):
    if getattr(cls, "__static_init__", None):
        cls.__static_init__()
    return cls
@static_init
class BB:
    NB_ROWS = 6
    NB_COLS = 7    

    # make us of a static constructor so that these fields get computed only once.
    @classmethod
    def __static_init__(BB):
        BB.SIZE = max(BB.NB_ROWS, BB.NB_COLS)
        BB.MASK_COL = sum([1 << i for i in range(BB.SIZE)])
        BB.MASK_FULL = sum([
            1 << (i + i // BB.NB_ROWS)
            for i in range(BB.NB_ROWS * BB.NB_COLS)
        ])

    def __init__(self, initial=0):
        self.bb = initial
    
    def __hash__(self):
        return self.bb

    def __int__(self):
        return self.bb

    def __eq__(self, other):
        return other == self.bb or (isinstance(other, BB) and self.bb == other.bb)

    def clear(self):
        self.bb = 0


    def __getitem__(self, key) :
        if isinstance(key, tuple) and len(key) == 2:
            return self.get(key[0], key[1])
        raise IndexError()

    def __setitem__(self, key, value) :
        if value != 1:
            raise Exception("Can only set 1 !")
        if isinstance(key, tuple) and len(key) == 2:
            return self.set(key[0], key[1])
        raise IndexError()

    def count(self):
        return BB.Count(self.bb)

    # STATIC METHODS
    def Count(bb:int):
        cnt = 0
        while bb != 0:
            bb &= bb-1
            cnt += 1
        return cnt
        
    def get(self, row:int, col:int) -> int:
        return BB.Get(self.bb, row, col)

    def Get(bb:int, row:int, col:int) -> int:
        index = row + BB.SIZE * col
        ret = ((bb >> index) & 1)
        return ret

    def set(self, row:int, col:int) -> int:
        self.bb = BB.Set(self.bb, row, col)
        return self.bb

    def Set(bb:int, row:int, col:int) -> int:
        if col >= BB.NB_COLS:
            raise Exception(
                f"There are only {BB.NB_COLS} columns, not {col+1} !")
        if row >= BB.NB_ROWS:
            raise Exception(
                f"There are only {BB.NB_ROWS} rows, not {row+1} !")
        index = row + BB.SIZE * col
        return bb | (1 << int(index))

    def getCol(self, col):
        return (self.bb >> (col * BB.SIZE)) & self.mask_col

    def getRow(self, row):
        return sum([(self.bb >> (row + BB.SIZE * i) & 1) << i
                    for i in range(BB.SIZE)])

    def printBB(self):
        print(BB.NB_ROWS, 'x', BB.NB_COLS, '=>', self.bb)
        for row in reversed(range(BB.NB_ROWS)):
            print([self.get(row, col) for col in range(BB.NB_COLS)])

    def hasFour(self):
        return BB.HasFour(self.bb)

    def HasFour(bb:int):
        # Check \
        temp_bboard = bb & (bb >> (BB.SIZE - 1))
        if (temp_bboard & (temp_bboard >> (2 * (BB.SIZE - 1)))):
            return True
        # Check -
        temp_bboard = bb & (bb >> BB.SIZE)
        if (temp_bboard & (temp_bboard >> 2 * BB.SIZE)):
            return True
        # Check /
        temp_bboard = bb & (bb >> (BB.SIZE + 1))
        if (temp_bboard & (temp_bboard >> 2 * (BB.SIZE + 1))):
            return True
        # Check |
        temp_bboard = bb & (bb >> 1)
        if (temp_bboard & (temp_bboard >> 2 * 1)):
            return True
        return False

    def isFull(self) -> bool:
        return BB.IsFull(self.bb)

    def IsFull(bb:int) -> bool:
        return (bb & BB.MASK_FULL) == BB.MASK_FULL

    def addToColumn(self, col):
        if col >= BB.NB_COLS:
            raise Exception(
                f"There are only {BB.NB_COLS} columns, not {col+1} !")
        for row in range(0, BB.NB_ROWS):
            if self.get(row, col) == 0:
                return self.set(row, col)