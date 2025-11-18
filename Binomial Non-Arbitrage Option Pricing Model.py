# https://medium.com/@xiucat/understanding-binomial-non-arbitrage-pricing-models-a-real-world-perspective-261ee64f7690
# https://gregorygundersen.com/blog/2023/06/03/binomial-options-pricing-model/

import numpy as np

def binomial_option_pricing(S, K, r, T, N, is_call=True):
	"""Calculate the option price using the binomial model.
	
	Arguments:
	S -- Current price of the underlying asset
	K -- Strike price of the option
	r -- Risk-free interest rate
	T -- Time to expiration in years
	N -- Number of time steps
	is_call -- Boolean indicating whether it is a call option (True) or a put option (False)
	
	Returns:
	Option price
	"""
	dt = T / N  # Time step size
	u = np.exp(r * dt)  # Upward factor
	d = 1 / u  # Downward factor
	p = (np.exp(r * dt) - d) / (u - d)  # Probability of upward movement

	# Initialize the price tree
	price_tree = np.zeros((N + 1, N + 1))
	price_tree[0, 0] = S

	# Calculate the price at each node of the tree
	for i in range(1, N + 1):
		price_tree[i, 0] = price_tree[i - 1, 0] * u
		for j in range(1, i + 1):
			price_tree[i, j] = price_tree[i - 1, j - 1] * d

	# Calculate the option value at each node of the tree
	option_tree = np.zeros((N + 1, N + 1))
	for j in range(N + 1):
		if is_call:
			option_tree[N, j] = max(price_tree[N, j] - K, 0)
		else:
			option_tree[N, j] = max(K - price_tree[N, j], 0)

	# Backward calculation of option prices
	for i in range(N - 1, -1, -1):
		for j in range(i + 1):
			option_tree[i, j] = np.exp(-r * dt) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])

	return option_tree[0, 0]


# Example usage
S = 100  # Current price of the underlying asset
K = 105  # Strike price
r = 0.05  # Risk-free interest rate
T = 1  # Time to expiration in years
N = 100  # Number of time steps
is_call = True  # Whether it is a call option (True) or a put option (False)

option_price = binomial_option_pricing(S, K, r, T, N, is_call)
print("Option Price:", option_price)
