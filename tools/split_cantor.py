import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the directory exists
os.makedirs('./images', exist_ok=True)

def float_to_bin(x, length):
    """Converts a float in [0,1] to a binary string of given length."""
    bits = []
    val = x
    for _ in range(length):
        val *= 2
        if val >= 1:
            bits.append('1')
            val -= 1
        else:
            bits.append('0')
    return "".join(bits)

def phi_w(x_vals, w):
    """
    Computes the Dyadic Prefix Comparator phi_w(x).
    Returns 1 if x|_{|w|} <= w, else 0.
    """
    w_len = len(w)
    y_vals = []
    
    # We define a lexicographical comparison function
    for x in x_vals:
        # Convert x to binary string of same length as w
        # Note: We add a tiny epsilon to handle floating point boundary issues cleanly for visualization
        x_bin = float_to_bin(x, w_len)
        
        if x_bin <= w:
            y_vals.append(1.0)
        else:
            y_vals.append(0.0)
    return np.array(y_vals)

def generate_split_cantor_plot():
    # Setup
    x = np.linspace(0, 1, 1000)
    target_a = 0.5  # We focus on the split at 0.5 (binary 0.1000...)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # Plot 1: Approximating f_a+ (The "<=" case)
    # Sequence: w converges to a from above/matches prefix
    # a = 1000... 
    # w sequence: 1, 10, 100, 1000 (Prefixes of 0.5)
    # ---------------------------------------------------------
    sequences_plus = ["1", "10", "100"] 
    colors_plus = ['#a1c9f4', '#8de5a1', '#ff9f9b']
    
    ax1 = axes[0]
    for i, w in enumerate(sequences_plus):
        y = phi_w(x, w)
        ax1.step(x, y, where='post', label=f'w="{w}"', color=colors_plus[i], linewidth=2)
        
    ax1.axvline(target_a, color='black', linestyle='--', alpha=0.3)
    ax1.set_title(r"Converging to $f_a^+$ ($x \leq 0.5$)" + "\n(Prefixes of $a$)")
    ax1.set_xlabel("Input Space (Interval [0,1])")
    ax1.set_ylabel("Output")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.2)

    # ---------------------------------------------------------
    # Plot 2: Approximating f_a- (The "<" case)
    # Sequence: w converges to a from below (Strictly smaller)
    # a = 1000...
    # w sequence: 0, 01, 011 (Predecessors)
    # ---------------------------------------------------------
    sequences_minus = ["0", "01", "011"]
    colors_minus = ['#d0bbff', '#debb9b', '#fab0e4']
    
    ax2 = axes[1]
    for i, w in enumerate(sequences_minus):
        y = phi_w(x, w)
        ax2.step(x, y, where='post', label=f'w="{w}"', color=colors_minus[i], linewidth=2)
        
    ax2.axvline(target_a, color='black', linestyle='--', alpha=0.3)
    ax2.set_title(r"Converging to $f_a^-$ ($x < 0.5$)" + "\n(Predecessors of $a$)")
    ax2.set_xlabel("Input Space (Interval [0,1])")
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.2)

    # ---------------------------------------------------------
    # Plot 3: The Split (Zoomed Logic)
    # Visualizing the difference at the specific point 0.5
    # ---------------------------------------------------------
    ax3 = axes[2]
    
    # Limit functions idealized
    # f_a+ is 1 at 0.5
    # f_a- is 0 at 0.5
    
    # Draw f_a+ (shifted up slightly for visibility)
    ax3.plot([0, 0.5], [1.02, 1.02], color='blue', linewidth=3, label=r'$f_a^+$ (Limit 1)')
    ax3.plot([0.5, 1.0], [0.02, 0.02], color='blue', linewidth=3)
    # Point at 0.5
    ax3.scatter([0.5], [1.02], color='blue', s=100, zorder=5)
    
    # Draw f_a- (shifted down slightly for visibility)
    ax3.plot([0, 0.5], [0.98, 0.98], color='red', linewidth=3, linestyle='--', label=r'$f_a^-$ (Limit 2)')
    ax3.plot([0.5, 1.0], [-0.02, -0.02], color='red', linewidth=3, linestyle='--')
    # Point at 0.5 (Empty circle implies value is 0, but step drops before)
    # Actually f_a-(0.5) = 0. So we put a solid dot at 0.
    ax3.scatter([0.5], [-0.02], color='red', s=100, zorder=5)

    ax3.set_xlim(0.4, 0.6)
    ax3.set_ylim(-0.1, 1.2)
    ax3.set_title("The Split at $x=0.5$")
    ax3.set_xlabel("Input Space")
    ax3.text(0.51, 1.05, r"Includes $x=0.5$", color='blue')
    ax3.text(0.51, -0.08, r"Excludes $x=0.5$", color='red')
    ax3.legend(loc='center right')
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('./images/split_cantor_diagram.png')
    print("Diagram saved to ./images/split_cantor_diagram.png")

if __name__ == "__main__":
    generate_split_cantor_plot()