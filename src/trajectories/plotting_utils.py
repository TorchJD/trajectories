"""Utility functions for plotting."""


def map_orders_to_indices(
    aggregator_keys: list[str], aggregator_order: dict[str, int]
) -> dict[str, int]:
    """Map aggregator keys to their indices based on sorted order.

    This function takes the available aggregators and maps them to sequential indices
    (0, 1, 2, ...) based on their order values. This ensures that subplots are positioned
    correctly regardless of which aggregators are actually present.

    Args:
        aggregator_keys: List of aggregator keys to map.
        aggregator_order: Dictionary mapping aggregator keys to their order values.

    Returns:
        Dictionary mapping aggregator keys to their indices (0-based).

    Example:
        If aggregator_keys = ["mean", "dualproj", "aligned_mtl"] with orders [0, 2, 8],
        this returns {"mean": 0, "dualproj": 1, "aligned_mtl": 2}.
    """
    # Sort aggregator keys by their order values
    sorted_keys = sorted(aggregator_keys, key=lambda k: aggregator_order[k])
    # Create mapping from key to index
    return {key: idx for idx, key in enumerate(sorted_keys)}


def compute_subplot_layout(n_aggregators: int) -> tuple[int, int]:
    """Compute subplot layout (n_rows, n_cols) based on number of aggregators.

    Args:
        n_aggregators: Number of aggregators to plot.

    Returns:
        A tuple of (n_rows, n_cols) for the subplot grid.

    Raises:
        ValueError: If n_aggregators is not between 1 and 10.
    """
    if n_aggregators <= 5:
        return 1, n_aggregators
    elif n_aggregators == 6:
        return 2, 3
    elif n_aggregators == 7:
        return 2, 4
    elif n_aggregators == 8:
        return 2, 4
    elif n_aggregators == 9:
        return 2, 5
    elif n_aggregators == 10:
        return 2, 5
    else:
        raise ValueError(f"Unsupported number of aggregators: {n_aggregators}")


def get_subplot_position(
    order: int, n_aggregators: int, n_rows: int, n_cols: int
) -> tuple[int, int]:
    """Convert order index to (row, col) position.

    Args:
        order: The order index of the aggregator.
        n_aggregators: Total number of aggregators.
        n_rows: Number of rows in the subplot grid.
        n_cols: Number of columns in the subplot grid.

    Returns:
        A tuple of (row, col) for the subplot position.
    """
    if n_rows == 1:
        return 0, order
    else:
        # For 2 rows
        if n_aggregators in [6, 8, 10]:
            # Even split
            row = order // (n_aggregators // 2)
            col = order % (n_aggregators // 2)
            return row, col
        elif n_aggregators in [7, 9]:
            # One more on first row
            first_row_count = (n_aggregators + 1) // 2
            if order < first_row_count:
                return 0, order
            else:
                return 1, order - first_row_count
        else:
            raise ValueError(
                f"Unsupported combination of n_aggregators={n_aggregators}, n_rows={n_rows}"
            )


def get_unused_subplot_positions(
    n_aggregators: int, n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    """Get list of unused subplot positions.

    For layouts where not all subplot positions are used (e.g., 7 aggregators in a 2x4 grid),
    this function returns the positions that should be hidden.

    Args:
        n_aggregators: Total number of aggregators.
        n_rows: Number of rows in the subplot grid.
        n_cols: Number of columns in the subplot grid.

    Returns:
        List of (row, col) tuples for unused positions.
    """
    if n_rows == 1:
        # Single row, all positions are used (n_cols == n_aggregators)
        return []

    # Compute all used positions
    used_positions = set()
    for idx in range(n_aggregators):
        pos = get_subplot_position(idx, n_aggregators, n_rows, n_cols)
        used_positions.add(pos)

    # Find all positions and return unused ones
    all_positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    return [pos for pos in all_positions if pos not in used_positions]
