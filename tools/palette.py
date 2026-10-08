"""Shared "bubblegum" colour palette for every figure in the paper.

All figure scripts import their colours from here, so changing a value
below recolours every illustration consistently.

The five colours were checked with a simulation of protanopia,
deuteranopia and tritanopia (Machado et al. 2009): every pair stays at
least 23 CIELAB units apart, so they remain distinguishable for
colour-blind readers.
"""

PINK = "#F25CA8"       # bubblegum pink
SKY = "#59D1E8"        # sky blue
LEMON = "#FFD84D"      # lemon
LAVENDER = "#A58BFF"   # lavender
BERRY = "#9C1C6E"      # berry

# Newton fractals: a purple palette of their own.  These five colours
# passed the same colour-blindness check (every pair at least 28 CIELAB
# units apart under protanopia, deuteranopia and tritanopia).
NEWTON_PURPLE = "#6A2DD8"   # royal purple
NEWTON_ORCHID = "#D65DB1"   # orchid
NEWTON_GOLD = "#F5B700"     # gold
NEWTON_LILAC = "#E6DAFF"    # pale lilac
NEWTON_PLUM = "#3B1259"     # deep plum

NEWTON_ROOT_COLORS = [NEWTON_PURPLE, NEWTON_ORCHID, NEWTON_GOLD]  # real, upper, lower root
NEWTON_CYCLE_COLORS = [NEWTON_LILAC, NEWTON_PLUM]                 # attracting cycle: near 0, near 1
NEWTON_SLOW_FADE = 0.25   # slowly converging points fade this far toward white

# Line drawings: main colour, accent for deep computations (limits),
# and a light-to-full ramp for successive iterates n.
MAIN = "#2BA9C9"       # a deeper sky blue, so thin lines stay visible
ACCENT = PINK
RAMP = ["#C4ECF5", "#8ED9EC", "#59C4DE", MAIN]
