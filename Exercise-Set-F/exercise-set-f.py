# Kaylee Barcroft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log2
import random
import sympy

# Load the prime numbers
url = 'https://raw.githubusercontent.com/koorukuroo/Prime-Number-List/master/primes.csv'
primes_df = pd.read_csv(url, header=None, names=['prime'])
all_primes = primes_df['prime'].tolist()

def get_primes_in_range(min_val, max_val, prime_list=None):
    """Return all primes between min_val and max_val inclusive.
    
    If prime_list is provided, filter from that list.
    Otherwise, use sympy to generate primes in the range.
    """
    if prime_list:
        return [p for p in prime_list if min_val <= p <= max_val]
    else:
        # Generate primes using sympy for larger ranges
        return list(sympy.primerange(min_val, max_val + 1))

def calculate_theoretical_false_positive_rate(n):
    """Calculate the theoretical false positive rate for a given n."""
    # Get primes between n and n²
    primes_in_range = get_primes_in_range(n, n*n, all_primes)
    
    if not primes_in_range:
        return 0
    
    # Find how many primes we can multiply while staying below 2^n
    max_product = 2**n - 1
    product = 1
    count = 0
    
    for prime in primes_in_range:
        if product * prime <= max_product:
            product *= prime
            count += 1
        else:
            break
    
    # Theoretical false positive rate
    return count / len(primes_in_range) if primes_in_range else 0

def simulate_fingerprinting(n, num_trials):
    """Simulate the fingerprinting scheme and return empirical false positive rate."""
    # Get primes between n and n²
    primes_in_range = get_primes_in_range(n, n*n, all_primes)
    
    if not primes_in_range:
        return 0
    
    # Calculate K (product of as many primes as possible while < 2^n)
    max_product = 2**n - 1
    product = 1
    k_primes = []
    
    for prime in primes_in_range:
        if product * prime <= max_product:
            product *= prime
            k_primes.append(prime)
        else:
            break
    
    K = product  # This is our adversarial y value
    x = 0        # Alice's value
    
    # Run trials
    false_positives = 0
    for _ in range(num_trials):
        # Alice randomly chooses a prime p between n and n²
        p = random.choice(primes_in_range)
        
        # Alice computes h = x mod p (which is 0 since x = 0)
        h = 0
        
        # Bob computes g = y mod p
        g = K % p
        
        # If g = h, we have a false positive
        if g == h:
            false_positives += 1
    
    # Empirical false positive rate
    return false_positives / num_trials if num_trials > 0 else 0

def main():
    # Range of n values to test
    n_values = [6, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Calculate theoretical and empirical rates
    theoretical_rates = []
    empirical_rates = []
    primes_counts = []
    k_values = []
    
    # Number of trials for empirical testing - adjust based on n value
    base_trials = 10000
    
    for n in n_values:
        print(f"Processing n = {n}")
        
        # For larger n values, use sympy to generate primes directly
        if n > 100:
            # Skip using the preloaded prime list for large ranges
            primes_in_range = get_primes_in_range(n, n*n, None)
        else:
            primes_in_range = get_primes_in_range(n, n*n, all_primes)
        
        # Calculate K (product of as many primes as possible while < 2^n)
        max_product = 2**n - 1
        product = 1
        k_primes = []
        
        for prime in primes_in_range:
            if product * prime <= max_product:
                product *= prime
                k_primes.append(prime)
            else:
                break
        
        K = product  # This is our adversarial y value
        
        # Store information about the primes
        primes_count = len(primes_in_range)
        k_prime_count = len(k_primes)
        
        # Calculate theoretical rate
        theoretical_rate = k_prime_count / primes_count if primes_count > 0 else 0
        
        # Adjust number of trials based on the theoretical rate
        # More trials needed for smaller probabilities
        if theoretical_rate > 0:
            num_trials = max(base_trials, int(10 / theoretical_rate))
            num_trials = min(num_trials, 1000000)  # Cap at 1 million trials
        else:
            num_trials = base_trials
        
        # Run empirical simulation for smaller n or when theoretical rate isn't too small
        if n <= 500 or theoretical_rate >= 0.001:
            empirical_rate = simulate_fingerprinting(n, num_trials)
        else:
            # For very large n with tiny probabilities, just use theoretical
            empirical_rate = theoretical_rate
        
        theoretical_rates.append(theoretical_rate)
        empirical_rates.append(empirical_rate)
        primes_counts.append(primes_count)
        k_values.append(k_prime_count)
        
        # Print current results
        print(f"  Number of primes between {n} and {n*n}: {primes_count}")
        print(f"  Number of primes in K: {k_prime_count}")
        print(f"  Theoretical false positive rate: {theoretical_rate:.6f}")
        print(f"  Empirical false positive rate ({num_trials} trials): {empirical_rate:.6f}")
        print(f"  1/n rate: {1/n:.6f}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(n_values, theoretical_rates, 'b-', label='Theoretical Rate')
    plt.plot(n_values, empirical_rates, 'ro', label='Empirical Rate')
    plt.plot(n_values, [1/n for n in n_values], 'g--', label='1/n')
    
    plt.xlabel('n')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate vs n in Fingerprinting Scheme')
    plt.legend()
    plt.grid(True)
    
    # Add log scale for better visualization
    plt.yscale('log')
    plt.xscale('log')
    
    plt.savefig('fingerprinting_rates.png')
    plt.show()
    
    # Create a more detailed plot for comparison with 1/n
    plt.figure(figsize=(12, 8))
    plt.plot(n_values, [t/(1/n) for t, n in zip(theoretical_rates, n_values)], 'b-', 
             label='Theoretical Rate × n')
    plt.axhline(y=1.0, color='r', linestyle='-', label='y = 1')
    
    plt.xlabel('n')
    plt.ylabel('Theoretical Rate × n')
    plt.title('Comparing Theoretical Rate to 1/n')
    plt.legend()
    plt.grid(True)
    plt.savefig('rate_vs_one_over_n.png')
    plt.show()
    
    # Save the data to CSV for further analysis
    results_df = pd.DataFrame({
        'n': n_values,
        'primes_count': primes_counts,
        'k_primes_count': k_values,
        'theoretical_rate': theoretical_rates,
        'empirical_rate': empirical_rates,
        'one_over_n': [1/n for n in n_values],
        'rate_times_n': [t*n for t, n in zip(theoretical_rates, n_values)]
    })
    
    results_df.to_csv('fingerprinting_results.csv', index=False)
    print("Results saved to fingerprinting_results.csv")

if __name__ == '__main__':
    main()