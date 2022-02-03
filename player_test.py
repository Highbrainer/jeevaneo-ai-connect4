import unittest
from bb import BB
from player import Player


class PlayerTest(unittest.TestCase):

    def test_children_number(self):
        bb = BB()
        self.assertEqual(7, len(Player.children(bb, BB())))
        for col in range(bb.nb_cols):
            for _ in range(bb.nb_rows):
                bb.addToColumn(col)
            self.assertEqual(bb.nb_cols - col - 1,
                             len(Player.children(bb, BB())))



    def test_children_full(self):
        bb1 = BB()
        bb2 = BB(initial=bb1.mask_full)
        self.assertEqual(0, len(Player.children(bb1, bb2)))

    def test_children(self):
        bb1 = BB(initial=1)
        bb2 = BB(initial=2)
        children = Player.children(bb1, bb2)
        self.assertEqual(7, len(children))
        self.assertEqual(0, children[0][0])
        self.assertEqual(5, children[0][1].bb)
        self.assertEqual(2, children[0][2].bb)
        for col in range(1, bb1.nb_cols):
            self.assertEqual(col, children[col][0])
            self.assertEqual(1, children[col][1].get(0, col))
            self.assertEqual(2, children[col][2].bb)



    
unittest.main(argv=['bidon'], exit=False)