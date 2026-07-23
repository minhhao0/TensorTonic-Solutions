def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    lr=0
    if step>=total_steps:
        return final_lr
    for i in range(0,total_steps):
        if i >step:
            break
        if i<warmup_steps:
            lr=(i*initial_lr)/warmup_steps
        elif i>=warmup_steps and i<=total_steps:
            lr=final_lr+(initial_lr-final_lr)*(total_steps-i)/(total_steps-warmup_steps)
    return lr