# Kaylee Barcroft
# Project 2 - Death Saves Probabilistic Study
# note: hp is normalized to simplify the calculations
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

d20 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # omega (possible outcomes)

def main():
    # Run experiments with different iteration counts
    iterations = [5, 10, 50, 100, 1000, 10000]
    
    # Run experiments without potion option
    print("Running experiments without potion option...")
    no_potion_results = {}
    for iter_count in iterations:
        hp_outcomes, roll_results = runExperiment(d20, iter_count, use_potion=False)
        no_potion_results[iter_count] = (hp_outcomes, roll_results)
        
        # Calculate statistics
        avg_hp = sum(hp_outcomes) / len(hp_outcomes)
        success_rate = sum(1 for hp in hp_outcomes if hp > 0) / len(hp_outcomes)
        full_recovery_rate = sum(1 for hp in hp_outcomes if hp == 0.5) / len(hp_outcomes)
        death_rate = sum(1 for hp in hp_outcomes if hp == 0) / len(hp_outcomes)
        
        print(f"\nIteration count: {iter_count}")
        print(f"Average HP outcome: {avg_hp:.4f}")
        print(f"Success rate: {success_rate:.4f}")
        print(f"Full recovery rate: {full_recovery_rate:.4f}")
        print(f"Death rate: {death_rate:.4f}")
    
    # Run experiments with potion option
    print("\nRunning experiments with potion option...")
    potion_results = {}
    for iter_count in iterations:
        hp_outcomes, roll_results = runExperiment(d20, iter_count, use_potion=True)
        potion_results[iter_count] = (hp_outcomes, roll_results)
        
        # Calculate statistics
        avg_hp = sum(hp_outcomes) / len(hp_outcomes)
        success_rate = sum(1 for hp in hp_outcomes if hp > 0) / len(hp_outcomes)
        full_recovery_rate = sum(1 for hp in hp_outcomes if hp == 0.5) / len(hp_outcomes)
        potion_rate = sum(1 for hp in hp_outcomes if hp == 0.25) / len(hp_outcomes)
        death_rate = sum(1 for hp in hp_outcomes if hp == 0) / len(hp_outcomes)
        
        print(f"\nIteration count: {iter_count}")
        print(f"Average HP outcome: {avg_hp:.4f}")
        print(f"Success rate: {success_rate:.4f}")
        print(f"Full recovery rate: {full_recovery_rate:.4f}")
        print(f"Potion rate: {potion_rate:.4f}")
        print(f"Death rate: {death_rate:.4f}")
    
    # Plot results
    plotResults(no_potion_results, potion_results)

def runExperiment(die, num_iterations, use_potion=False):
    """Run multiple death save simulations and return the results"""
    hp_outcomes = []
    all_results = []
    
    for _ in range(num_iterations):
        hp, results = rollSaves(die, use_potion)
        hp_outcomes.append(hp)
        all_results.append(results)
    
    return hp_outcomes, all_results

def rollSaves(die, use_potion=False):
    """Simulate a single death save sequence"""
    rolling = True
    rollPass = 0
    rollFail = 0
    results = []
    
    while rolling:
        result = rollDice(die)
        results.append(result)
        
        if use_potion and rollPass > 0:
            # Calculate expected value based on current state
            potion_value = 0.25
            expected_value = expVal([rollPass, rollFail], die)
            
            if potion_value >= expected_value:
                hp = 0.25
                results.append('potion')
                rolling = False
                continue
        
        if result == 1:  # nat 1 = insta-die
            hp = 0
            rolling = False
        elif result == 20:  # nat 20 = insta-get-up
            hp = 0.5
            rolling = False
        elif result < 10:  # one fail
            rollFail += 1
            if rollFail == 3:
                hp = 0
                rolling = False
        elif result >= 10:  # one success
            rollPass += 1
            if rollPass == 3:
                hp = 0.5
                rolling = False
    
    return hp, results

def expVal(results, die):
    """Calculate expected value of continuing to roll based on current state"""
    success_count = results[0]
    fail_count = results[1]
    
    # Probability of getting exactly 3 successes given current state
    p_success = 0
    
    # Handle different states
    if success_count == 0:
        # Need 3 more successes, and can have 0-2 more failures
        p_success = ((11/20) ** 3) * (1 - (9/20) ** 3)
    elif success_count == 1:
        # Need 2 more successes, and can have 0-2 more failures (adjusted for fail_count)
        remaining_fails = 2 - fail_count
        p_success = ((11/20) ** 2) * (1 - (9/20) ** (remaining_fails + 1))
    elif success_count == 2:
        # Need 1 more success, and can have 0-2 more failures (adjusted for fail_count)
        remaining_fails = 2 - fail_count
        p_success = (11/20) * (1 - (9/20) ** (remaining_fails + 1))
    
    # Expected value is probability of success times the reward (0.5)
    expected_value = p_success * 0.5
    
    return expected_value

def rollDice(die):
    """Roll a single die and return the result"""
    return np.random.choice(die)

def plotResults(no_potion_results, potion_results):
    """Plot the results of the experiments"""
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Death Saves Simulation Results', fontsize=16)
    
    # Extract the highest iteration count for detailed analysis
    max_iter = max(no_potion_results.keys())
    
    # Subplot 1: HP distribution without potion
    hp_no_potion = no_potion_results[max_iter][0]
    hp_counts = Counter(hp_no_potion)
    labels = ['Death (0)', 'Stabilized (0.5)']
    sizes = [hp_counts.get(0, 0), hp_counts.get(0.5, 0)]
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title(f'Outcome Distribution (No Potion, n={max_iter})')
    
    # Subplot 2: HP distribution with potion
    hp_with_potion = potion_results[max_iter][0]
    hp_counts = Counter(hp_with_potion)
    labels = ['Death (0)', 'Potion (0.25)', 'Stabilized (0.5)']
    sizes = [hp_counts.get(0, 0), hp_counts.get(0.25, 0), hp_counts.get(0.5, 0)]
    axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title(f'Outcome Distribution (With Potion, n={max_iter})')
    
    # Subplot 3: Average HP by iteration count
    iterations = list(no_potion_results.keys())
    avg_hp_no_potion = [sum(no_potion_results[i][0])/len(no_potion_results[i][0]) for i in iterations]
    avg_hp_with_potion = [sum(potion_results[i][0])/len(potion_results[i][0]) for i in iterations]
    
    axes[1, 0].plot(iterations, avg_hp_no_potion, 'bo-', label='Without Potion')
    axes[1, 0].plot(iterations, avg_hp_with_potion, 'ro-', label='With Potion')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Number of Iterations')
    axes[1, 0].set_ylabel('Average HP Outcome')
    axes[1, 0].set_title('Average HP by Iteration Count')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Subplot 4: Number of rolls histogram
    roll_counts_no_potion = [len(results) for results in no_potion_results[max_iter][1]]
    roll_counts_with_potion = [len(results) for results in potion_results[max_iter][1]]
    
    axes[1, 1].hist(roll_counts_no_potion, alpha=0.5, label='Without Potion', bins=range(1, 7))
    axes[1, 1].hist(roll_counts_with_potion, alpha=0.5, label='With Potion', bins=range(1, 7))
    axes[1, 1].set_xlabel('Number of Rolls')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Rolls Until Resolution')
    axes[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('death_saves_results.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()