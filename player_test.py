import unittest
from bb import BB
from player import Player
import time
import math

class PlayerTest(unittest.TestCase):

    def test_children_number(self):
        bb = BB()
        self.assertEqual(7, len(Player.children(bb.bb, 0, is_player_one=True)))
        for col in range(BB.NB_COLS):
            for _ in range(BB.NB_ROWS):
                bb.addToColumn(col)
            self.assertEqual(BB.NB_COLS - col - 1,
                             len(Player.children(bb.bb, 0, is_player_one=True)))



    def test_children_full(self):
        bb1 = BB()
        bb2 = BB(initial=BB.MASK_FULL)
        self.assertEqual(0, len(Player.children(bb1.bb, bb2.bb, is_player_one=True)))

    def test_children(self):
        bb1 = BB(initial=1)
        bb2 = BB(initial=2)
        children = Player.children(bb1.bb, bb2.bb, is_player_one=True)
        self.assertEqual(7, len(children))
        self.assertEqual(0, children[0][0])
        self.assertEqual(5, children[0][1])
        self.assertEqual(2, children[0][2])
        for col in range(1, BB.NB_COLS):
            self.assertEqual(col, children[col][0])
            self.assertEqual(1, BB.Get(children[col][1], 0, col))
            self.assertEqual(2, children[col][2])

    def test_alphaBeta(self):
        
        bb1 = BB()
        bb2 = BB()
        me = Player(5, True)
        bb1_b = BB()
        bb2_b = BB()
        me_b = Player(5, True)

        start = time.time()
        res = Player.alphaBeta(bb1.bb, bb2.bb, me.depthLimit, me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        start2 = time.time()        
        res2 = Player.alphaBeta(bb1_b.bb, bb2_b.bb, me_b.depthLimit, me_b.isPlayerOne, -math.inf, math.inf)
        print("Result :", res2)
        end2 = time.time()
        print("Took2", end2 - start2, "s.")

        bb1.set(0, 0)

        res = Player.alphaBeta(bb1.bb, bb2.bb, me.depthLimit, not me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        bb2.set(0, 1)

        res = Player.alphaBeta(bb1.bb, bb2.bb, me.depthLimit, me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        bb1.set(1, 0)

        res = Player.alphaBeta(bb1.bb, bb2.bb, me.depthLimit, not me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        bb2.set(1, 1)

        res = Player.alphaBeta(bb1.bb, bb2.bb, me.depthLimit, me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)
        end = time.time()
        print("Took", end - start, "s.")

        print(Player.alphaBeta.cache_info())

    
unittest.main(argv=['bidon'], exit=False)