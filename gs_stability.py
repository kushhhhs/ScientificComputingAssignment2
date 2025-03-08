"""This model checks if a gray scott model is time dependent or not"""

def detect_pattern_variance(gs, iterations=5000, check_interval=100, threshold=1e-5):
    """Determines stability based on the variance of U over time.
    """
    prev_var = np.var(gs.U[1:-1, 1:-1])
    for i in range(iterations // check_interval):
        for _ in range(check_interval):
            gs.update()
        
        current_var = np.var(gs.U[1:-1, 1:-1])

        # If the variance stabilizes, the pattern is stable
        if abs(current_var - prev_var) < threshold:
            return "Stable"
        
        prev_var = current_var

    return "Time-Dependent"


def detect_pattern_mean(gs, iterations=5000, check_interval=100, threshold=1e-5):
    """Determines whether the gs reaches a stable pattern or time-dependent.
    """
    # Save initial state
    prev_U = np.copy(gs.U[1:-1, 1:-1])
    for i in range(iterations // check_interval):
        for _ in range(check_interval):
            gs.update()

        #Compute mean squared difference
        diff = np.mean((gs.U[1:-1, 1:-1] - prev_U) ** 2)

        # If changes bellow threshold, assume stability
        if diff < threshold:
            return "Stable"

        prev_U = np.copy(gs.U[1:-1, 1:-1])

    return "Time-Dependent"
