# Kaylee Barcroft
# Project 2 - Death Saves Probabilistic Study
# note: hp is normalized to simplify the calculations

import numpy as np
import matplotlib.pyplot as plt

d20 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # omega (possible outcomes)


def main():
    runExperiment(d20)

def runExperiment(die):
    iterations = [5,10,20,50,100]
    hp, results = rollSaves(die)
    print(f'results: {results}')


# rollSaves() is the roll simulator for death saves. It takes the die we're rolling
# and returns the hp value (reward value) obtained from the simulation as well as the 
# values that were rolled/decisions that were made.
def rollSaves(die):
    rolling = True
    rollPass = 0
    rollFail = 0

    potion = 0 #.25

    while rolling:
        result = rollDice(die)
        print(f'result: {result}')

        results = [rollPass, rollFail]


        if potion >= expVal(results, die): # check if the expected value of potion >= roll
            hp = 0.25
            results.append('potion')
            rolling = False

        elif result == 1: # nat 1 = insta-die
            hp = 0
            rolling = False

        elif result == 20: # nat 20 = insta-get-up
            hp = 0.5
            rolling = False

        elif result < 10: # one fail
            rollFail += 1
            if rollFail == 3:
                hp = 0
                rolling = False

        elif result >= 10: # one success
            rollPass += 1
            if rollPass == 3:
                hp = 0.5
                rolling = False

    return hp, [rollPass, rollFail]
        

def expVal(results, die):
    # determine current game state
    stateSuccess = results[0]
    stateFail = results[1]

    if stateSuccess == 2:
        succVal = ((9/20) * (1/2))
    
    if stateFail == 2:
        failVal = ((9/20) * 0)

    if die == d20:
        critFailVal = (1/20 * 0)
        critSuccVal = ((1/20) * (1/2))

        value = critFailVal + critSuccVal + ((9/20) * 0) + ((9/20) * (1/2))
    
    return value


def rollDice(die):
    return np.random.choice(die)


def plotResults(hpOutcomes, results):
    pass

if __name__ == '__main__':
    main()