def discretize_action(discretized_actions, sampled_action):
    """
    Map a sampled action to the index of the discretized action.
    Args:
    - sampled_action: The action sampled from the environment's action space.
    - discretized_actions: Array of action medians representing each interval.
    Returns:
    - Index of the discretized action.
    """
    interval_length = 4.0 / len(discretized_actions)  # Total range divided by number of actions
    # Find the index by calculating how many interval lengths the sampled action is from the start (-2)
    action_index = int((sampled_action + 2) / interval_length)
    # Clamp the index to be within valid range
    action_index = max(0, min(action_index, len(discretized_actions) - 1))
    return action_index