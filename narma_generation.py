#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def generate_narma_from_input(u_normalized, narma_order):
    """
    Generate NARMA-n time series from normalized input.
    
    Args:
        u_normalized: Normalized input [0, 0.5]
        narma_order: NARMA order
    
    Returns:
        y_narma: NARMA output series
    """
    n_samples = len(u_normalized)
    y_narma = np.zeros(n_samples)
    
    if narma_order == 2:
        for k in range(1, n_samples - 1):
            y_narma[k + 1] = (0.4 * y_narma[k] + 
                            0.4 * y_narma[k] * y_narma[k - 1] + 
                            0.6 * u_normalized[k]**3 + 0.1)
                            
    elif narma_order == 3:
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(2, n_samples - 1):
            y_sum = np.sum(y_narma[k - 2:k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - 2] * u_normalized[k] + d)
                            
    else:
        # NARMA-n for n >= 4
        n = narma_order
        a, b, c, d = 0.3, 0.05, 1.5, 0.1
        for k in range(n - 1, n_samples - 1):
            y_sum = np.sum(y_narma[k - (n - 1):k + 1])
            y_narma[k + 1] = (a * y_narma[k] + 
                            b * y_narma[k] * y_sum + 
                            c * u_normalized[k - n + 1] * u_normalized[k] + d)
    
    return y_narma
#%%
def u(t, period_ratio=2, amplitude=0.2):
    """Input signal: sum of three sinusoids"""
    f1 = 2.11
    f2 = 3.73
    f3 = 4.33
    T = period_ratio
    u_val = amplitude * np.sin(2*np.pi * f1 * t / T) * np.sin(2*np.pi * f2 * t / T) * np.sin(2*np.pi * f3 * t / T)
    return u_val

#%%
dt = 0.01
u_raw = np.array([u(t, period_ratio=1, amplitude=0.5) for t in np.arange(10000)*dt])
u_normalized = u_raw - u_raw.min()
u_normalized = 0.5 * u_normalized / u_normalized.max()
plt.plot(u_raw[:1000])
plt.show()
plt.plot(u_normalized[:1000])
plt.show()
# %%
y_narma_2 = generate_narma_from_input(u_normalized, narma_order=2)
y_narma_3 = generate_narma_from_input(u_normalized, narma_order=3)
y_narma_5 = generate_narma_from_input(u_normalized, narma_order=5)
plt.plot(y_narma_2[5:100], label='NARMA-2')
plt.plot(y_narma_3[5:100], label='NARMA-3')
plt.plot(y_narma_5[5:100], label='NARMA-5')
plt.legend()
plt.show()
#%%
y_narma_2 = generate_narma_from_input(u_raw, narma_order=2)
y_narma_3 = generate_narma_from_input(u_raw, narma_order=3)
y_narma_5 = generate_narma_from_input(u_raw, narma_order=5)
plt.plot(y_narma_2[5:5000], label='NARMA-2')
plt.plot(y_narma_3[5:5000], label='NARMA-3')
plt.plot(y_narma_5[5:5000], label='NARMA-5')
plt.legend()
plt.show()
# %%
