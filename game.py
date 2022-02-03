from player import Player, ManualPlayer
from env import MyPuissance4Env
from env import REWARD


if __name__ == "__main__":
    env = MyPuissance4Env()
    players = [Player(2, True), ManualPlayer()]

    while(True):
        for player in players :
            action = player.findMove(env)
            time_step = env.step(action)
            print(env.bb_players[0].bb, env.bb_players[1].bb)
            env.print_bb()

        # is the game over ?
        reward = time_step.reward
        if reward == REWARD.DRAW:
            print("It's a draw!")
            break
        elif reward == REWARD.WIN:
            print("Player 1 wins!")
            break
        elif reward == REWARD.LOST:
            print("Player 2 wins!")
            break
