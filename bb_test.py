import unittest
from bb import BB


class BBTest(unittest.TestCase):

    def test_hasFour_not(self):
        bb = BB()
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(0)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(4)
        self.assertEqual(False, bb.hasFour())

    def test_hasFour_horizontal(self):
        bb = BB()
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(0)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(True, bb.hasFour())
        bb.addToColumn(3)
        self.assertEqual(True, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(True, bb.hasFour())

    def test_hasFour_vertical(self):
        bb = BB()
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(0)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(True, bb.hasFour())

    def test_hasFour_diag_up(self):
        bb = BB()
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(1)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(2)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(3)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(3)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(3)
        self.assertEqual(False, bb.hasFour())
        bb.addToColumn(3)
        self.assertEqual(True, bb.hasFour())

    def test_hasFour_dup2(self):
        bb1 = BB(initial=2223359239)
        self.assertTrue(bb1.hasFour())

    def test_hasFour_diag_down(self):
        bb = BB()
        self.assertEqual(False, bb.hasFour())
        bb.set(0, 5)
        self.assertEqual(False, bb.hasFour())
        bb.set(1, 4)
        self.assertEqual(False, bb.hasFour())
        bb.set(2, 3)
        self.assertEqual(False, bb.hasFour())
        bb.set(3, 2)
        self.assertEqual(True, bb.hasFour())
        bb.set(3, 1)
        self.assertEqual(True, bb.hasFour())
        bb.set(4, 0)
        self.assertEqual(True, bb.hasFour())

    def test_isFull(self):
        bb = BB()
        for row in range(bb.nb_rows):
            for col in range(bb.nb_cols):
                self.assertFalse(bb.isFull())
                bb.set(row, col)
        self.assertTrue(bb.isFull())

    def test_count(self):
        bb = BB()
        self.assertEqual(0, bb.count())
        bb.set(0, 5)
        self.assertEqual(1, bb.count())
        bb.set(1, 4)
        self.assertEqual(2, bb.count())
        bb.set(2, 3)
        self.assertEqual(3, bb.count())
        bb.set(3, 2)
        self.assertEqual(4, bb.count())
        bb.set(3, 1)
        self.assertEqual(5, bb.count())
        bb.set(4, 0)
        self.assertEqual(6, bb.count())


    def test_hash(self):
        bb1 = BB(initial=2223359239)
        self.assertEqual(bb1.bb, hash(bb1))

    def test_equal(self):
        bb1 = BB(initial=12)
        bb2 = BB(initial=12)
        bb3 = BB(initial=13)

        self.assertEqual(bb1, bb1)
        self.assertEqual(bb1, bb2)
        self.assertEqual(bb1, 12)
        self.assertEqual(12, bb2)
        self.assertNotEqual(bb1, bb3)
        self.assertNotEqual(bb1, 1)
        self.assertNotEqual(1, bb1)

unittest.main(argv=['bidon'], exit=False)