import math

from numpy import empty, uint

from bb import BB
from env import MyPuissance4Env

from functools import lru_cache

# ### additional imports from plotting
# import matplotlib.pyplot as plt
# from igraph import Graph
# ###

DEBUGGING = False


def debug(msg):
    if DEBUGGING:
        print(msg)


# ### debug graph stuff
# class GNode:
#   def __init__(self, next:int, score:int, bb1:BB, bb2:BB, action:int=None):
#     self.action=action
#     self.score=score
#     self.bb1=bb1
#     self.bb2=bb2
#     self.next=next

#   # networkx fails if str() gets special characters... keep it simple.
#   def __str__(self):
#      return str(self.__hash__())

#   def label(self):
#     return f"a:{self.action}\nn:{self.next}\ns:{self.score}"


#   def image(self):
#       env = MyPuissance4Env()
#       env.bb_players[0]=self.bb1
#       env.bb_players[0]=self.bb2
#       img = env.render()
#       env.close()
#       return img
# ###
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
            #debug("ESTIMATION: DRAW !")
            bb_current.printBB()
            return Player.ESTIMATE_DRAW

        #debug(f'Player 1 ({bb1.bb}):')
        estimation_p1 = Player.estimate_player(bb1, bb2)
        #debug(f'Player 2 ({bb2.bb}):')
        estimation_p2 = -Player.estimate_player(bb2, bb1)
        estimation = estimation_p1 + estimation_p2
        #debug(f'ESTIMATION: P1:{estimation_p1}\tP2:{estimation_p2}\tGLOBAL:{estimation}\t')
        return estimation

    def estimate_player(bb_estimated: BB, bb_other: BB) -> int:
        # 4
        if bb_estimated.hasFour():
            #debug(f' Found 4 !')
            return Player.ESTIMATE_4

        estimation = 0

        empty = ~(bb_estimated.bb | bb_other.bb)

        # three vertically, with an empty cell above
        # player 1
        vthrees = ((bb_estimated.bb & bb_estimated.bb << 1) &
                   ((bb_estimated.bb & bb_estimated.bb << 1) << 1))
        vthrees_with_empty_above = (vthrees << 1) & empty
        nb_vthrees = BB(initial=vthrees_with_empty_above).count()
        #debug(f' Found {nb_vthrees} vertical 3s')
        estimation += nb_vthrees * Player.ESTIMATE_3

        # two vertically, with an empty cell above
        # player 1
        vtwos = (bb_estimated.bb & bb_estimated.bb << 1) & ~vthrees
        vtwos_with_empty_above = (vtwos << 1) & empty
        nb_vtwos = BB(initial=vtwos_with_empty_above).count()
        #debug(f' Found {nb_vtwos} vertical 2s')
        estimation += nb_vtwos * Player.ESTIMATE_2

        SHIFT = bb_estimated.size

        # three horizontally
        hthrees = ((bb_estimated.bb & bb_estimated.bb << SHIFT) &
                   ((bb_estimated.bb & bb_estimated.bb << SHIFT) << SHIFT))
        # three with a free slot on their right
        hthrees_with_empty_right = (hthrees << SHIFT) & empty
        nb_hthrees_r = BB(initial=hthrees_with_empty_right).count()
        #debug(f' Found {nb_hthrees_r} horizontal 3s with right free space')
        estimation += nb_hthrees_r * Player.ESTIMATE_3
        # three with a free slot on their left
        hthrees_with_empty_left = (hthrees >> (2 * SHIFT)) & empty
        nb_hthrees_l = BB(initial=hthrees_with_empty_left).count()
        #debug(f' Found {nb_hthrees_l} horizontal 3s with left free space')
        estimation += nb_hthrees_l * Player.ESTIMATE_3

        # two horizontally
        htwos = bb_estimated.bb & (bb_estimated.bb << SHIFT) & ~hthrees
        # two with a free slot on their right
        htwos_with_empty_right = ((htwos) << SHIFT) & empty
        nb_htwos_r = BB(initial=htwos_with_empty_right).count()
        #debug(f' Found {nb_htwos_r} horizontal 2s with right free space')
        estimation += nb_htwos_r * Player.ESTIMATE_2
        # two with a free slot on their left
        htwos_with_empty_left = (htwos >> (2 * SHIFT)) & empty
        nb_htwos_l = BB(initial=htwos_with_empty_left).count()
        #debug(f' Found {nb_htwos_l} horizontal 2s with left free space')
        estimation += nb_htwos_l * Player.ESTIMATE_2

        # one
        ones = bb_estimated.bb & (
            bb_estimated.bb << SHIFT) & ~htwos & ~vtwos & ~hthrees & ~vthrees
        # one with a free slot above
        ones_with_empty_above = (ones << 1) & empty
        nb_ones_a = BB(initial=ones_with_empty_above).count()
        #debug(f' Found {nb_ones_a} slots with a free space above')
        estimation += nb_ones_a * Player.ESTIMATE_1

        # one with a free slot on their right
        ones_with_empty_right = (ones << SHIFT) & empty
        nb_ones_r = BB(initial=ones_with_empty_right).count()
        #debug(f' Found {nb_ones_r} slots with a free space on the right')
        estimation += nb_ones_r * Player.ESTIMATE_1

        # one with a free slot on their left
        ones_with_empty_left = (ones >> SHIFT) & empty
        nb_ones_l = BB(initial=ones_with_empty_left).count()

        estimation += nb_ones_l * Player.ESTIMATE_1
        #debug(f' Found {nb_ones_l} slots with a free space on the left')

        # diag down,left>up,right
        dupthrees = (bb_estimated.bb & (bb_estimated.bb <<
                                        (SHIFT + 1))) & ((bb_estimated.bb &
                                                          (bb_estimated.bb <<
                                                           (SHIFT + 1))) <<
                                                         (SHIFT + 1))
        # diag up with a free fourth slot on their upper right
        dupthrees_with_empty_up_right = (dupthrees << (SHIFT + 1)) & empty
        nb_dup_r = BB(initial=dupthrees_with_empty_up_right).count()
        #debug(f' Found {nb_dup_r} up right diags with a free space on the top right')
        estimation += nb_dup_r * Player.ESTIMATE_3

        # diag up with a free fourth slot on their bottom left
        dupthrees_with_empty_bottom_left = (dupthrees >>
                                            (3 * (SHIFT + 1))) & empty
        nb_dup_l = BB(initial=dupthrees_with_empty_bottom_left).count()
        #debug(f' Found {nb_dup_l} up right diags with a free space on the bottom left')
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
        nb_ddown_r = BB(initial=ddownthrees_with_empty_bottom_right).count()
        #debug(f' Found {nb_ddown_r} down left diags with a free space on the bottom right')
        estimation += nb_ddown_r * Player.ESTIMATE_3

        # diag down with a free fourth slot on their upper left
        ddownthrees_with_empty_up_left = (ddownthrees >>
                                          (3 * (SHIFT - 1))) & empty
        nb_ddown_l = BB(initial=ddownthrees_with_empty_up_left).count()
        #debug(f' Found {nb_ddown_l} down left diags with a free space on the upper left')
        estimation += nb_ddown_l * Player.ESTIMATE_3

        return estimation

    # returns the optimal column to move in by implementing the Alpha-Beta algorithm
    def findMove(self, env: MyPuissance4Env):
        # ### debug graph stuff
        # self.G = Graph()
        # ###
        score, move, root = Player.alphaBeta(env.bb_players[0],
                                           env.bb_players[1], self.depthLimit,
                                           self.isPlayerOne, -math.inf,
                                           math.inf)
        ### debug graph stuff
        #showGraph(self.G, root_name=str(root), title=f'Step {env.current_step}')
        # pos = graphviz_layout(self.G, prog="dot")
        # fig, ax = plt.subplots()
        # nx.draw(self.G, pos, ax=ax, with_labels=not True, font_weight='bold', font_size=16, node_size=3000 )

        # labels = { o:o.label() for o in self.G.nodes}
        # nx.draw_networkx_labels(self.G, pos, labels)

        # # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
        # tr_figure = ax.transData.transform
        # # Transform from display to figure coordinates
        # tr_axes = fig.transFigure.inverted().transform

        # # Select the size of the image (relative to the X axis)
        # print("X axis : ", ax.get_xlim())
        # icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
        # icon_center = icon_size / 2.

        # if False:
        #   for n in self.G.nodes:
        #     xf, yf = tr_figure(pos[n])
        #     xa, ya = tr_axes((xf, yf))
        #     # get overlapped axes and plot icon
        #     x = xa - icon_center
        #     y = ya - icon_center
        #     #a = plt.axes([x, y, icon_size, icon_size])
        #     a = plt.axes([x, y, 1, 1])
        #     print(f'...computing next image... {n.label()} with size {icon_size} at coord {x},{y}')
        #     a.imshow(n.image())
        #     a.axis("off")

        # plt.axis("off")
        # plt.show()
        ###

        return move

    def children(bb1: BB, bb2: BB, is_player_one: bool):
        ret = []
        bb_current = BB(initial=bb1.bb | bb2.bb)

        child_bb1 = bb1
        child_bb2 = bb2

        if is_player_one:
            bb = bb1
        else:
            bb = bb2

        for col in range(bb.nb_cols):
            for row in range(bb.nb_rows):
                if bb_current.get(row, col) == 0:
                    child = BB(initial=bb.bb)
                    child.set(row, col)
                    if is_player_one:
                        child_bb1 = child
                    else:
                        child_bb2 = child
                    ret.append((col, child_bb1, child_bb2))
                    break
        return ret

    def indent(self, depth: int):
        ret = ''
        for _ in range(depth, self.depthLimit):
            ret += '  '
        return ret

    # findMove helper function, utilizing alpha-beta pruning within the  minimax algorithm
    @lru_cache(maxsize=32*1024*1024)
    def alphaBeta(bb1: BB, bb2: BB, depth, player, alpha, beta):
        #debug(f'{self.indent(depth)}alphaBeta({bb1.bb}, {bb2.bb}, {depth}, {player}, {alpha}, {beta})')

        bb_current = BB(initial=bb1.bb | bb2.bb)
        game_over = False
        if bb_current.isFull():
            estimation = -math.inf if player else math.inf
            #debug(f'{self.indent(depth)}returns DRAW {estimation}, {-2}')
            game_over = True
        elif depth == 0:
            estimation = Player.estimate(bb1, bb2)
            #debug(f'{self.indent(depth)}returns LEAF {estimation}, {-1}')
            game_over = True
        elif bb1.hasFour() or bb2.hasFour(
        ):  # could optimize by checking only one based on boolean player
            estimation = Player.estimate(bb1, bb2)
            #debug(f'{self.indent(depth)}returns GAME OVER {estimation}, {-1000}')
            game_over = True

        ### debug graph stuff
        g_me = None
        ###

        if game_over:
            # ### debug graph stuff
            # g_me = GNode(score=estimation, next=-1, bb1=bb1, bb2=bb2)
            # self.G.add_vertex(name=str(g_me), node=g_me)
            # ###

            return estimation, -1, g_me

        if player:
            bestScore = -math.inf
            shouldReplace = lambda x: x > bestScore
        else:
            bestScore = math.inf
            shouldReplace = lambda x: x < bestScore

        bestMove = -1

        children = Player.children(bb1, bb2, player)
        # ### debug graph stuff
        # g_children = []
        # ###
        #debug(f'{self.indent(depth)}Found {len(children)} children')
        for i, child in enumerate(children):
            #debug(f'{self.indent(depth-1)}Child {i}')
            action, child_bb1, child_bb2 = child
            #debug(f"{self.indent(depth-1)}if player{1 if player else 2} plays {action}...")
            temp, bm, g_child = Player.alphaBeta(child_bb1, child_bb2, depth - 1,
                                               not player, alpha, beta)
            if not g_child is None:
                g_child.action = action
            if shouldReplace(temp):
                bestScore = temp
                bestMove = action
            # ### debug graph stuff
            # g_children.append(g_child)
            # ###
            if player:
                alpha = max(alpha, temp)
            else:
                beta = min(beta, temp)
            if alpha >= beta:
                break
        #debug(f'{self.indent(depth)}returns estimated {bestScore}, {bestMove}')

        # ### debug graph stuff
        # g_me = GNode(score=bestScore, next=bestMove, bb1=bb1, bb2=bb2)
        # self.G.add_vertex(name=str(g_me), node=g_me)
        # for g_child in g_children:
        #     self.G.add_edge(str(g_me), str(g_child))
        # ###

        return bestScore, bestMove, g_me


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
