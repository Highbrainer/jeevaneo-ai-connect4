import unittest
from bb import BB
from player import Player
import time
import math

class PlayerTest(unittest.TestCase):

    def test_children_number(self):
        bb = BB()
        self.assertEqual(7, len(Player.children(bb, BB(), is_player_one=True)))
        for col in range(bb.nb_cols):
            for _ in range(bb.nb_rows):
                bb.addToColumn(col)
            self.assertEqual(bb.nb_cols - col - 1,
                             len(Player.children(bb, BB(), is_player_one=True)))



    def test_children_full(self):
        bb1 = BB()
        bb2 = BB(initial=bb1.mask_full)
        self.assertEqual(0, len(Player.children(bb1, bb2, is_player_one=True)))

    def test_children(self):
        bb1 = BB(initial=1)
        bb2 = BB(initial=2)
        children = Player.children(bb1, bb2, is_player_one=True)
        self.assertEqual(7, len(children))
        self.assertEqual(0, children[0][0])
        self.assertEqual(5, children[0][1].bb)
        self.assertEqual(2, children[0][2].bb)
        for col in range(1, bb1.nb_cols):
            self.assertEqual(col, children[col][0])
            self.assertEqual(1, children[col][1].get(0, col))
            self.assertEqual(2, children[col][2].bb)

    def test_alphaBeta(self):
        
        bb1 = BB()
        bb2 = BB()
        me = Player(5, True)
        bb1_b = BB()
        bb2_b = BB()
        me_b = Player(5, True)

        start = time.time()
        res = Player.alphaBeta(bb1, bb2, me.depthLimit, me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        start2 = time.time()        
        res2 = Player.alphaBeta(bb1_b, bb2_b, me_b.depthLimit, me_b.isPlayerOne, -math.inf, math.inf)
        print("Result :", res2)
        end2 = time.time()
        print("Took2", end2 - start2, "s.")

        bb1.set(0, 0)

        res = Player.alphaBeta(bb1, bb2, me.depthLimit, not me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        bb2.set(0, 1)

        res = Player.alphaBeta(bb1, bb2, me.depthLimit, me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        bb1.set(1, 0)

        res = Player.alphaBeta(bb1, bb2, me.depthLimit, not me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)

        bb2.set(1, 1)

        res = Player.alphaBeta(bb1, bb2, me.depthLimit, me.isPlayerOne, -math.inf, math.inf)
        print("Result :", res)
        end = time.time()
        print("Took", end - start, "s.")

        print(Player.alphaBeta.cache_info())

    
unittest.main(argv=['bidon'], exit=False)