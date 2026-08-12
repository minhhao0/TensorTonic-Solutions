import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    # Your implementation here
    generated_samples=np.array(generated_samples)
    std=generated_samples.std(axis=0)
    diversity_score=std.mean()
    is_collapsed=diversity_score<threshold
    return {
        "diversity_score":diversity_score,
        "is_collapsed":is_collapsed
    }
    