# Kaylee Barcroft
# Project 2 - Death Saves Probabilistic Study
# note: hp is normalized to simplify the calculations

import numpy as np

d20 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # omega (possible outcomes)


def main():

    rollSaves(d20)


# rollSaves() is the roll simulator for death saves. It takes the die we're rolling
# and returns the hp value (reward value) obtained from the simulation as well as the 
# values that were rolled/decisions that were made.
def rollSaves(die):
    rolling = True
    rollPass = 0
    rollFail = 0
    results = []
    potion = 0.25
    while rolling:
        result = rollDice(die)
        results.append(result)
        print(f'result: {result}')
        if potion >= expVal(results): # check if the expected value of potion >= roll
            hp = 0.25
            results.append('potion')
            rolling = False
        elif result == 1:
            hp = 0
            rolling = False
        elif result == 20:
            hp = 0.5
            rolling = False
        elif result < 10:
            rollFail += 1
            if rollFail == 3:
                hp = 0
                rolling = False
        elif result >= 10:
            rollPass += 1
            if rollPass == 3:
                hp = 0.5
                rolling = False
    return hp, results
        

def expVal(state):
    value = 0
    
    return value


def rollDice(die):
    return np.random.choice(die)

if __name__ == '__main__':
    main()