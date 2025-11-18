import math

points = {
    'P1': (3, 7), 'P2': (4, 6), 'P3': (5, 5), 'P4': (6, 4),
    'P5': (7, 3), 'P6': (6, 2), 'P7': (7, 2), 'P8': (8, 4),
    'P9': (3, 3), 'P10': (2, 6), 'P11': (3, 5), 'P12': (2, 4)
}

eps = 1.9
mx = 4

def dis(a, b):
    return math.dist(a, b)
    
def findpadosi(point):
    return [
        q for q in points
        if q != point and dis(points[point], points[q]) <= eps
    ]

# Step 1: Identify all core points first
core_points = set()
for p in points:
    if len(findpadosi(p)) + 1 >= mx:
        core_points.add(p)

# Step 2: Identify border points
border_points = set()
for p in points:
    if p not in core_points:
        if any(n in core_points for n in findpadosi(p)):
            border_points.add(p)

# Step 3: Remaining → noise
noise_points = set(points) - core_points - border_points

# Step 4: print results
for p in points:
    if p in core_points:
        print(p, "→ core")
    elif p in border_points:
        print(p, "→ border")
    else:
        print(p, "→ noise")
