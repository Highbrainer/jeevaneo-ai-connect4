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

    def __init__(self, depthLimit: uint, isPlayerOne: bool):

        self.isPlayerOne = isPlayerOne
        self.depthLimit = depthLimit

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
            estimation = MyPuissance4Env.estimate(bb1, bb2)
            #debug(f'{self.indent(depth)}returns LEAF {estimation}, {-1}')
            game_over = True
        elif bb1.hasFour() or bb2.hasFour(
        ):  # could optimize by checking only one based on boolean player
            estimation = MyPuissance4Env.estimate(bb1, bb2)
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
