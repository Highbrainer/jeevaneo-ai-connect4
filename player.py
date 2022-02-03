import math

from numpy import empty, uint

from bb import BB
from env import MyPuissance4Env

#from board import Board

DEBUGGING = False

def debug(msg):
    if DEBUGGING:
        print(msg)

class Player:

    ESTIMATE_4 = 100000
    ESTIMATE_3 = 1000
    ESTIMATE_2 = 10
    ESTIMATE_1 = 1
    ESTIMATE_DRAW = 0

    def __init__(self, depthLimit: uint, isPlayerOne: bool):

        self.isPlayerOne = isPlayerOne
        self.depthLimit = depthLimit

    def estimate(bb1: BB, bb2: BB) -> int:

        bb_current = BB(initial=bb1.bb | bb2.bb)
        # draw
        if (bb_current.isFull()):
            debug("ESTIMATION: DRAW !")
            bb_current.printBB()
            return Player.ESTIMATE_DRAW

        debug(f'Player 1 ({bb1.bb}):')
        estimation = Player.estimate_player(bb1, bb2)
        debug(f'Player 2 ({bb2.bb}):')
        estimation -= Player.estimate_player(bb2, bb1)
        debug(f'ESTIMATION: {estimation}')
        return estimation

    def estimate_player(bb_estimated: BB, bb_other: BB) -> int:
        # 4
        if bb_estimated.hasFour():            
            debug(f' Found 4 !')
            return Player.ESTIMATE_4

        estimation = 0

        empty = ~(bb_estimated.bb | bb_other.bb)

        # three vertically, with an empty cell above
        # player 1
        vthrees = ((bb_estimated.bb & bb_estimated.bb << 1) &
                   ((bb_estimated.bb & bb_estimated.bb << 1) << 1))
        vthrees_with_empty_above = (vthrees << 1) & empty
        nb_vthrees = BB(
            initial=vthrees_with_empty_above).count()
        debug(f' Found {nb_vthrees} vertical 3s')
        estimation += nb_vthrees * Player.ESTIMATE_3

        # two vertically, with an empty cell above
        # player 1
        vtwos = (bb_estimated.bb & bb_estimated.bb << 1) & ~vthrees
        vtwos_with_empty_above = (vtwos << 1) & empty
        nb_vtwos = BB(
            initial=vtwos_with_empty_above).count()
        debug(f' Found {nb_vtwos} vertical 2s')
        estimation += nb_vtwos * Player.ESTIMATE_2

        SHIFT = bb_estimated.size

        # three horizontally
        hthrees = ((bb_estimated.bb & bb_estimated.bb << SHIFT) &
                   ((bb_estimated.bb & bb_estimated.bb << SHIFT) << SHIFT))
        # three with a free slot on their right
        hthrees_with_empty_right = (hthrees << SHIFT) & empty
        nb_hthrees_r = BB(
            initial=hthrees_with_empty_right).count()
        debug(f' Found {nb_hthrees_r} horizontal 3s with right free space')
        estimation += nb_hthrees_r * Player.ESTIMATE_3
        # three with a free slot on their left
        hthrees_with_empty_left = (hthrees >> (2 * SHIFT)) & empty
        nb_hthrees_l = BB(
            initial=hthrees_with_empty_left).count()
        debug(f' Found {nb_hthrees_l} horizontal 3s with left free space')
        estimation += nb_hthrees_l * Player.ESTIMATE_3

        # two horizontally
        htwos = bb_estimated.bb & (bb_estimated.bb << SHIFT) & ~hthrees
        # two with a free slot on their right
        htwos_with_empty_right = ((htwos) << SHIFT) & empty
        nb_htwos_r = BB(
            initial=htwos_with_empty_right).count()
        debug(f' Found {nb_htwos_r} horizontal 2s with right free space')
        estimation += nb_htwos_r * Player.ESTIMATE_2
        # two with a free slot on their left
        htwos_with_empty_left = (htwos >> (2 * SHIFT)) & empty
        nb_htwos_l = BB(
            initial=htwos_with_empty_left).count()
        debug(f' Found {nb_htwos_l} horizontal 2s with left free space')
        estimation += nb_htwos_l * Player.ESTIMATE_2

        # one
        ones = bb_estimated.bb & (
            bb_estimated.bb << SHIFT) & ~htwos & ~vtwos & ~hthrees & ~vthrees
        # one with a free slot above
        ones_with_empty_above = (ones << 1) & empty
        nb_ones_a = BB(
            initial=ones_with_empty_above).count()
        debug(f' Found {nb_ones_a} slots with a free space above')
        estimation += nb_ones_a * Player.ESTIMATE_1

        # one with a free slot on their right
        ones_with_empty_right = (ones << SHIFT) & empty
        nb_ones_r = BB(
            initial=ones_with_empty_right).count()
        debug(f' Found {nb_ones_r} slots with a free space on the right')
        estimation += nb_ones_r * Player.ESTIMATE_1

        # one with a free slot on their left
        ones_with_empty_left = (ones >> SHIFT) & empty
        nb_ones_l = BB(
            initial=ones_with_empty_left).count()

        estimation += nb_ones_l * Player.ESTIMATE_1
        debug(f' Found {nb_ones_l} slots with a free space on the left')

        # diag down,left>up,right
        dupthrees = (bb_estimated.bb & (bb_estimated.bb <<
                                        (SHIFT + 1))) & ((bb_estimated.bb &
                                                          (bb_estimated.bb <<
                                                           (SHIFT + 1))) <<
                                                         (SHIFT + 1))
        # diag up with a free fourth slot on their upper right
        dupthrees_with_empty_up_right = (dupthrees << (SHIFT + 1)) & empty
        nb_dup_r = BB(
            initial=dupthrees_with_empty_up_right).count()
        debug(f' Found {nb_dup_r} up right diags with a free space on the top right')
        estimation += nb_dup_r * Player.ESTIMATE_3

        # diag up with a free fourth slot on their bottom left
        dupthrees_with_empty_bottom_left = (dupthrees >>
                                            (3 * (SHIFT + 1))) & empty
        nb_dup_l = BB(initial=dupthrees_with_empty_bottom_left).count(
        )
        debug(f' Found {nb_dup_l} up right diags with a free space on the bottom left')
        estimation += nb_dup_l * Player.ESTIMATE_3

        # diag up,left>bottom,right
        ddownthrees = (bb_estimated.bb & (bb_estimated.bb <<
                                          (SHIFT - 1))) & ((bb_estimated.bb &
                                                            (bb_estimated.bb <<
                                                             (SHIFT - 1))) <<
                                                           (SHIFT - 1))
        # diag down with a free fourth slot on their bottom right
        ddownthrees_with_empty_bottom_right = (ddownthrees <<
                                               (SHIFT - 1)) & empty
        nb_ddown_r = BB(initial=ddownthrees_with_empty_bottom_right).count(
        )
        debug(f' Found {nb_ddown_r} down left diags with a free space on the bottom right')
        estimation += nb_ddown_r * Player.ESTIMATE_3

        # diag down with a free fourth slot on their upper left
        ddownthrees_with_empty_up_left = (ddownthrees >>
                                          (3 * (SHIFT - 1))) & empty
        nb_ddown_l = BB(
            initial=ddownthrees_with_empty_up_left).count()
        debug(f' Found {nb_ddown_l} down left diags with a free space on the upper left')
        estimation += nb_ddown_l * Player.ESTIMATE_3

        return estimation

    # returns the optimal column to move in by implementing the Alpha-Beta algorithm
    def findMove(self, env: MyPuissance4Env):
        score, move = self.alphaBeta(env.bb_players[0], env.bb_players[1],
                                     self.depthLimit, self.isPlayerOne,
                                     -math.inf, math.inf)
        return move

    def children(bb1: BB, bb2: BB):
        ret = []
        bb_current = BB(initial=bb1.bb | bb2.bb)
        for col in range(bb1.nb_cols):
            for row in range(bb1.nb_rows):
                if bb_current.get(row, col) == 0:
                    child = BB(initial=bb1.bb)
                    child.set(row, col)
                    ret.append((col, child, bb2))
                    break
        return ret

    def indent(self, depth: int):
        ret = ''
        for _ in range(depth, self.depthLimit):
            ret += '  '
        return ret


    # findMove helper function, utilizing alpha-beta pruning within the  minimax algorithm
    def alphaBeta(self, bb1: BB, bb2: BB, depth, player, alpha, beta):
        debug(
            f'{self.indent(depth)}alphaBeta({bb1.bb}, {bb2.bb}, {depth}, {player}, {alpha}, {beta}'
        )
        # bb1 = bb_a if player else bb_b
        # bb2 = bb_b if player else bb_a
        bb_current = BB(initial=bb1.bb | bb2.bb)
        if bb_current.isFull():
            estimation = -math.inf if player else math.inf
            debug(f'{self.indent(depth)}returns DRAW {estimation}, {-1}')
            return estimation, -1
        elif depth == 0:
            estimation = Player.estimate(bb1, bb2)
            debug(f'{self.indent(depth)}returns LEAF {estimation}, {-1}')
            return estimation, -1
        elif bb1.hasFour() or bb2.hasFour(): # could optimize by checking only one based on boolean player
            estimation = Player.estimate(bb1, bb2)
            debug(f'{self.indent(depth)}returns GAME OVER {estimation}, {-1}')
            return estimation, -1

        if player:
            bestScore = -math.inf
            shouldReplace = lambda x: x > bestScore
        else:
            bestScore = math.inf
            shouldReplace = lambda x: x < bestScore

        bestMove = -1

        children = Player.children(bb1, bb2)
        debug(f'{self.indent(depth)}Found {len(children)} children')
        for i, child in enumerate(children):
            debug(f'{self.indent(depth-1)}Child {i}')
            action, child_bb1, child_bb2 = child
            temp = self.alphaBeta(child_bb1, child_bb2, depth - 1, not player,
                                  alpha, beta)[0]
            if shouldReplace(temp):
                bestScore = temp
                bestMove = action
            if player:
                alpha = max(alpha, temp)
            else:
                beta = min(beta, temp)
            if alpha >= beta:
                break
        debug(f'{self.indent(depth)}returns estimated {bestScore}, {bestMove}')
        return bestScore, bestMove


class ManualPlayer():

    def findMove(self, env: MyPuissance4Env):
        opts = " "
        for c in range(env.nb_cols):
            opts += " " + (str(c + 1)
                           if env._isColumnAlreadyFull(c) else ' ') + "  "
        print(opts)

        col = input("Place a token in column: ")
        col = int(col) - 1
        return col
