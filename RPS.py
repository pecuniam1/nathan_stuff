import numpyLayers
from numpyLayers import ann__init__, ann_activate, ann_calc_grads, ann_add_grads, ann_monitor
import numpy as np
num_rounds, stop_score, num_players, end_trigger, player1_pts, player2_pts = 0, 0, 0, 0, 0, 0


rps_trans = {
    "rock": np.array([1, 0, 0], dtype='float64'),
    "paper": np.array([0, 1, 0], dtype='float64'),
    "scissors": np.array([0, 0, 1], dtype='float64')
}


rps_backwards = {
    '0': "Rock",
    '1': "Paper",
    '2': "Scissors"
}


def instructions():
    global num_rounds, stop_score, num_players, end_trigger
    num_rounds, stop_score = 0, 0  # defaults to prevent possible calling errors
    end_trigger = input(f"Select an type of end trigger:\n"
                        f"Number of [r]ounds\n"
                        f"First to a [s]core\n"
                        f"> ")
    if end_trigger in ['r', 'R']:  # stops after certain number of rounds
        end_trigger = 'rounds'
        num_rounds = int(input(f"How many rounds?\n"
                               f"> "))
    elif end_trigger in ['s', 'S']:  # stops after a player reaches a designated score
        end_trigger = 'score'
        stop_score = int(input(f"What score should stop the game?\n"
                               f"> "))
    else:  # invalid entry
        print("Invalid input\n\n")
        instructions()


def is_win(player1, player2, bool=False):
    i = 0
    for keys in list(rps_trans.values()):
        check = (keys + player1) == 2
        if any(check):
            if bool:
                return all(list(rps_trans.values())[i - 1] == list(player2))
            else:
                return 2 * float(all(list(rps_trans.values())[i - 1] == list(player2))) - 1
        i += 1


def play():
    # initiating settings and AI
    global player1_pts, player2_pts
    instructions()
    ANN = []
    ann__init__(ANN, num_layers=5, first_len=4, hidden_len=10, last_len=3)
    ANN[0].activ[:3], ANN[0].activ[3] = rps_trans["rock"], 1  # default first input

    if end_trigger == "rounds":
        for round_ind in range(num_rounds):
            ann_activate(ANN)
            comp_choice = list(rps_trans.values())[list(ANN[-1].activ).index(max(ANN[-1].activ))]  # chooses the first, highest output
            player_choice = rps_trans[input(f"[Rock], [Paper], or [Scissors]?\n> ")]
            ANN[-1].desired = player_choice
            ann_calc_grads(ANN)
            ANN[0].activ[-1] = is_win(player_choice, comp_choice)
            ann_add_grads(ANN, batch_size=1)  # batch size temporary
            print(f"\nComputer chose {rps_backwards[str(list(comp_choice).index(1))]}")
            if is_win(player_choice, comp_choice, True):
                print("You won! :)")
                player1_pts += 1
            else:
                print("You lost! :(")
                player2_pts += 1
    elif end_trigger == "score":
        while player1_pts < stop_score and player2_pts < stop_score:
            ann_activate(ANN)
            comp_choice = list(rps_trans.values())[list(ANN[-1].activ).index(max(ANN[-1].activ))]  # chooses the first, highest output
            player_choice = rps_trans[input(f"[Rock], [Paper], or [Scissors]?\n> ")]
            ANN[-1].desired = player_choice
            ann_monitor(ANN)
            ann_calc_grads(ANN)
            ANN[0].activ[-1] = is_win(player_choice, comp_choice)
            ann_add_grads(ANN, batch_size=2)  # batch size temporary
            print(f"\nComputer chose {rps_backwards[str(list(comp_choice).index(1))]}")
            if is_win(player_choice, comp_choice, True):
                print("You won! :)")
                player1_pts += 1
            else:
                print("You lost! :(")
                player2_pts += 1
    print(f"Player 1 score: {player1_pts}\n"
          f"Player 2 score: {player2_pts}")


play()
