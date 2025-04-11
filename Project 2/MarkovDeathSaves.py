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
    
    # Initialize expected value lookup table
    ev_cache = {}
    for s in range(3):
        for f in range(3):
            ev_cache[(s, f)] = calculate_expected_value(s, f)
    
    print("Expected values for different states (success, fail):")
    for state, ev in ev_cache.items():
        print(f"State {state}: EV = {ev:.4f}")
    
    # Run experiments without potion option
    print("\nRunning experiments without potion option...")
    no_potion_results = {}
    for iter_count in iterations:
        hp_outcomes, roll_results = runExperiment(d20, iter_count, use_potion=False, ev_cache=ev_cache)
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
        hp_outcomes, roll_results = runExperiment(d20, iter_count, use_potion=True, ev_cache=ev_cache)
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
    
    # Calculate theoretical probabilities
    theoretical_no_potion = calculate_expected_value(0, 0)
    print(f"\nTheoretical expected value (no potion): {theoretical_no_potion:.4f}")
    
    # Plot results
    plotResults(no_potion_results, potion_results, theoretical_no_potion)

def runExperiment(die, num_iterations, use_potion=False, ev_cache=None):
    """Run multiple death save simulations and return the results"""
    hp_outcomes = []
    all_results = []
    
    for _ in range(num_iterations):
        hp, results = rollSaves(die, use_potion, ev_cache)
        hp_outcomes.append(hp)
        all_results.append(results)
    
    return hp_outcomes, all_results

def rollSaves(die, use_potion=False, ev_cache=None):
    """Simulate a single death save sequence"""
    rolling = True
    rollPass = 0
    rollFail = 0
    roll_results = []  # Track dice roll outcomes
    decision_results = []  # Track decisions (including potion)
    
    while rolling:
        result = rollDice(die)
        roll_results.append(result)  # Add this roll to results
        
        if use_potion and rollPass > 0:  # Only consider potion after at least one success
            # Get expected value from cache based on current state
            current_state = (rollPass, rollFail)
            if current_state in ev_cache:
                expected_value = ev_cache[current_state]
                potion_value = 0.25
                
                if potion_value > expected_value:
                    hp = 0.25
                    decision_results.append('potion')
                    rolling = False
                    break
        
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
    
    # Combine all results (both rolls and decisions)
    combined_results = roll_results + decision_results
    
    return hp, combined_results

def calculate_expected_value(success_count, fail_count):
    """
    Calculate the expected value (average HP outcome) of continuing from the current state.
    Uses a dynamic programming approach to solve the Markov decision process.
    
    Args:
        success_count (int): Current number of successful death saves (0-2)
        fail_count (int): Current number of failed death saves (0-2)
    
    Returns:
        float: Expected HP value if continuing from this state
    """
    # Terminal states
    if success_count >= 3:
        return 0.5  # Stabilized
    if fail_count >= 3:
        return 0.0  # Dead
    
    # Use memoization to avoid recalculating states
    memo = {}
    
    def dp(s, f):
        """Recursive dynamic programming function with memoization"""
        # Check memo first
        if (s, f) in memo:
            return memo[(s, f)]
        
        # Terminal conditions
        if s >= 3:
            return 0.5
        if f >= 3:
            return 0.0
        
        # Calculate expected value based on possible dice outcomes
        # 1. Natural 1 (critical failure) - immediate death
        p_crit_fail = 1/20
        ev_crit_fail = 0.0
        
        # 2. Natural 20 (critical success) - immediate stabilization
        p_crit_success = 1/20
        ev_crit_success = 0.5
        
        # 3. Roll 2-9 (regular failure)
        p_reg_fail = 8/20
        ev_reg_fail = dp(s, f+1)
        
        # 4. Roll 10-19 (regular success)
        p_reg_success = 10/20
        ev_reg_success = dp(s+1, f)
        
        # Combine all possibilities for expected value
        ev = (p_crit_fail * ev_crit_fail +
              p_crit_success * ev_crit_success +
              p_reg_fail * ev_reg_fail +
              p_reg_success * ev_reg_success)
        
        # Store in memo
        memo[(s, f)] = ev
        return ev
    
    # Calculate expected value from current state
    return dp(success_count, fail_count)

def rollDice(die, weights=None):
    """
    Roll a single die and return the result
    
    Args:
        die (list): List of possible outcomes
        weights (list, optional): List of weights for each outcome. Defaults to None (equal weights).
    
    Returns:
        int: The result of the roll
    """
    return np.random.choice(die, p=weights)

def plotResults(no_potion_results, potion_results, theoretical_value):
    """Plot the results of the experiments"""
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
    axes[1, 0].axhline(y=theoretical_value, color='g', linestyle='--', label=f'Theoretical EV: {theoretical_value:.4f}')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Number of Iterations')
    axes[1, 0].set_ylabel('Average HP Outcome')
    axes[1, 0].set_title('Average HP by Iteration Count')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Subplot 4: Number of rolls histogram - FIXED THIS PART
    # Count only numeric values in the results (exclude 'potion' strings)
    roll_counts_no_potion = []
    for results in no_potion_results[max_iter][1]:
        # Count only integer values (dice rolls)
        roll_count = sum(1 for r in results if isinstance(r, (int, np.int64, np.int32)))
        roll_counts_no_potion.append(roll_count)
    
    roll_counts_with_potion = []
    for results in potion_results[max_iter][1]:
        # Count only integer values (dice rolls)
        roll_count = sum(1 for r in results if isinstance(r, (int, np.int64, np.int32)))
        roll_counts_with_potion.append(roll_count)
    
    # Determine the maximum number of rolls across all simulations for bin sizing
    max_rolls = max(max(roll_counts_no_potion, default=0), max(roll_counts_with_potion, default=0))
    bins = range(1, max_rolls + 2)  # +2 to ensure the last bin is included
    
    axes[1, 1].hist(roll_counts_no_potion, alpha=0.5, label='Without Potion', bins=bins)
    axes[1, 1].hist(roll_counts_with_potion, alpha=0.5, label='With Potion', bins=bins)
    axes[1, 1].set_xlabel('Number of Rolls')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Rolls Until Resolution')
    axes[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('death_saves_results.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()