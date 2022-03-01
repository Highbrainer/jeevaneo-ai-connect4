from player import Player, ManualPlayer
from env import MyPuissance4Env
from env import REWARD


def main():
    env = MyPuissance4Env()
    players = [ManualPlayer(), Player(7, not True), ]

    MID = 1
    LAST = 2

    current_player = 0

    time_step = env.reset()
    while time_step.step_type <= MID:
        player = players[current_player % 2]
        current_player += 1
        action = player.findMove(time_step)
        time_step = env.step(action)
        env.print_bb()
        env.render().show()
        if time_step.step_type == LAST:
            print("Game over")
            break
    print("Player ", ((current_player-1)%2)+1, " wins!")


if __name__ == "__main__":
    main()
