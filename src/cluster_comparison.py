import math
import itertools
from typing import List, Set, Any, Callable


# --- Shared Utility ---

def _get_all_items(C1: List[Set[Any]], C2: List[Set[Any]]) -> Set[Any]:
    """Utility to get all unique items across both clusterings."""
    items = set()
    for cluster_list in [C1, C2]:
        for cluster in cluster_list:
            items.update(cluster)
    return items


# --- Omega Index (Adjusted Rand for Overlapping) ---

def _get_co_cluster_set(C: List[Set[Any]], items_list: List[Any]) -> Set[tuple]:
    """Gets all pairs of items that appear together in at least one cluster."""
    co_cluster_set = set()
    pairs = list(itertools.combinations(items_list, 2))
    for cluster in C:
        for (i, j) in pairs:
            if i in cluster and j in cluster:
                co_cluster_set.add((i, j))
    return co_cluster_set


def omega_index(C1: List[Set[Any]], C2: List[Set[Any]]) -> float:
    """
    Calculates the Omega Index (Adjusted Rand Index for overlapping pairs).
    """
    items = list(_get_all_items(C1, C2))
    N = len(items)
    if N < 2:
        return 1.0  # Trivial case

    M = (N * (N - 1)) / 2  # Total number of pairs

    A = _get_co_cluster_set(C1, items)
    B = _get_co_cluster_set(C2, items)

    n_A = len(A)
    n_B = len(B)
    n_11 = len(A.intersection(B))  # Co-clustered in both

    if n_A == M and n_B == M:
        return 1.0  # Both are trivial (one big cluster)

    exp_index = (n_A * n_B) / M
    max_index = (n_A + n_B) / 2

    numerator = n_11 - exp_index
    denominator = max_index - exp_index

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return numerator / denominator


# --- Overlapping NMI (LFK Implementation) ---

def _nmi_safe_log2(x: float) -> float:
    return 0.0 if x <= 0 else math.log2(x)


def _nmi_h_binary(x: float) -> float:
    """Binary entropy function h(x)."""
    if x == 0 or x == 1:
        return 0.0
    return -x * _nmi_safe_log2(x) - (1 - x) * _nmi_safe_log2(1 - x)


def _nmi_H_joint(xi: float, yj: float, pij: float) -> float:
    """Joint entropy H(Xi, Yj)"""
    p11 = pij
    p10 = xi - pij
    p01 = yj - pij
    p00 = 1 - p11 - p10 - p01

    return - (_nmi_safe_log2(p00) * p00 +
              _nmi_safe_log2(p01) * p01 +
              _nmi_safe_log2(p10) * p10 +
              _nmi_safe_log2(p11) * p11)


def _nmi_H_unconditional(C: List[Set[Any]], N: float) -> float:
    """Calculates the total unconditional entropy H(C)"""
    if not C:
        return 0.0
    total_entropy = 0.0
    for cluster in C:
        x = len(cluster) / N
        total_entropy += _nmi_h_binary(x)
    return total_entropy / len(C)


def _nmi_H_conditional_sum(C1: List[Set[Any]], C2: List[Set[Any]], N: float) -> float:
    """Calculates the sum of conditional entropies, sum(H(Xi|Y))."""
    if not C1 or not C2:
        return 0.0

    total_cond_entropy = 0.0
    for c1 in C1:
        xi = len(c1) / N
        min_h_c1_c2 = 1.0  # Max possible entropy is 1

        for c2 in C2:
            yj = len(c2) / N
            pij = len(c1.intersection(c2)) / N

            h_c2 = _nmi_h_binary(yj)
            h_c1_c2_joint = _nmi_H_joint(xi, yj, pij)

            # H(C1|C2) = H(C1, C2) - H(C2)
            h_cond = h_c1_c2_joint - h_c2
            if h_cond < min_h_c1_c2:
                min_h_c1_c2 = h_cond

        total_cond_entropy += min_h_c1_c2

    return total_cond_entropy / len(C1)


def overlapping_nmi(C1: List[Set[Any]], C2: List[Set[Any]]) -> float:
    """
    Calculates the Overlapping Normalized Mutual Information (LFK implementation).
    """
    items = _get_all_items(C1, C2)
    N = float(len(items))
    if N == 0:
        return 1.0

    H_C1 = _nmi_H_unconditional(C1, N)
    H_C2 = _nmi_H_unconditional(C2, N)

    if H_C1 == 0 and H_C2 == 0:
        return 1.0

    H_C1_given_C2 = _nmi_H_conditional_sum(C1, C2, N)
    H_C2_given_C1 = _nmi_H_conditional_sum(C2, C1, N)

    I_C1_C2 = H_C1 - H_C1_given_C2
    I_C2_C1 = H_C2 - H_C2_given_C1

    numerator = I_C1_C2 + I_C2_C1
    denominator = H_C1 + H_C2

    if denominator == 0:
        return 1.0  # Should be covered by H_C1==0 and H_C2==0 check

    return numerator / denominator


# --- Set-Matching Metrics (F-Measure / Jaccard) ---

def _f1_score(c1: Set[Any], c2: Set[Any]) -> float:
    """Calculates the F1-Score between two sets."""
    len_c1 = len(c1)
    len_c2 = len(c2)

    if len_c1 == 0 or len_c2 == 0:
        return 0.0

    intersect = len(c1.intersection(c2))

    precision = intersect / len_c2
    recall = intersect / len_c1

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def _jaccard_index(c1: Set[Any], c2: Set[Any]) -> float:
    """Calculates the Jaccard Index between two sets."""
    intersect = len(c1.intersection(c2))
    union = len(c1.union(c2))

    if union == 0:
        return 1.0  # Both sets are empty, considered identical

    return intersect / union


def _asymmetric_set_similarity(
        C_A: List[Set[Any]],
        C_B: List[Set[Any]],
        sim_func: Callable[[Set[Any], Set[Any]], float]
) -> float:
    """
    Calculates the asymmetric similarity from C_A to C_B.

    For each cluster in C_A, find its best match in C_B, then
    compute the weighted average of these best-match scores.
    """
    total_weight = 0.0
    total_sim = 0.0

    if not C_A:
        return 1.0 if not C_B else 0.0  # Empty set matches empty set

    if not C_B:
        return 0.0  # No clusters in A can be matched

    for c_a in C_A:
        weight = len(c_a)
        if weight == 0:
            continue  # Do not score empty clusters

        total_weight += weight

        # Find the best matching cluster in C_B
        max_sim = 0.0
        for c_b in C_B:
            max_sim = max(max_sim, sim_func(c_a, c_b))

        total_sim += weight * max_sim

    if total_weight == 0:
        return 1.0  # Only contained empty clusters, trivially perfect

    return total_sim / total_weight


def _symmetric_set_similarity(
        C1: List[Set[Any]],
        C2: List[Set[Any]],
        sim_func: Callable[[Set[Any], Set[Any]], float]
) -> float:
    """Calculates the symmetric similarity between C1 and C2."""

    # If both are empty (no items), they are identical
    if not _get_all_items(C1, C2):
        return 1.0

    sim_12 = _asymmetric_set_similarity(C1, C2, sim_func)
    sim_21 = _asymmetric_set_similarity(C2, C1, sim_func)

    return (sim_12 + sim_21) / 2.0


def overlapping_f1_score(C1: List[Set[Any]], C2: List[Set[Any]]) -> float:
    """
    Calculates the symmetric Overlapping F1-Score.

    This metric measures how well each cluster in C1 is reproduced
    in C2, and vice-versa, using the F1-Score as the similarity
    measure between individual clusters.
    """
    return _symmetric_set_similarity(C1, C2, _f1_score)


def overlapping_jaccard_index(C1: List[Set[Any]], C2: List[Set[Any]]) -> float:
    """
    Calculates the symmetric Overlapping Jaccard Index.

    This metric measures how well each cluster in C1 is reproduced
    in C2, and vice-versa, using the Jaccard Index as the similarity
    measure between individual clusters.
    """
    return _symmetric_set_similarity(C1, C2, _jaccard_index)


if __name__ == '__main__':
    # Example Usage:

    # Non-overlapping (partition)
    C1_part = [set([1, 2]), set([3, 4])]
    C2_part = [set([1, 2]), set([3, 4])]  # Identical
    C3_part = [set([1, 3]), set([2, 4])]  # Different

    print("--- Partition Examples ---")
    print(f"Omega (Identical):     {omega_index(C1_part, C2_part):.4f}")
    print(f"NMI (Identical):       {overlapping_nmi(C1_part, C2_part):.4f}")
    print(f"F1 (Identical):        {overlapping_f1_score(C1_part, C2_part):.4f}")
    print(f"Jaccard (Identical):   {overlapping_jaccard_index(C1_part, C2_part):.4f}")
    print("---")
    print(f"Omega (Different):     {omega_index(C1_part, C3_part):.4f}")
    print(f"NMI (Different):       {overlapping_nmi(C1_part, C3_part):.4f}")  # NMI is 0 for partitions
    print(f"F1 (Different):        {overlapping_f1_score(C1_part, C3_part):.4f}")
    print(f"Jaccard (Different):   {overlapping_jaccard_index(C1_part, C3_part):.4f}")

    # Overlapping
    C1_over = [{1, 2, 3}, {3, 4, 5}]
    C2_over = [{1, 2, 3}, {3, 4, 5}]  # Identical
    C3_over = [{1, 2}, {4, 5}, {3}]  # Finer-grained
    C4_over = [{1, 2, 3, 4, 5}]  # Coarser-grained
    C5_over = [{1}, {4, 5}]  # Missing

    print("\n--- Overlapping Examples ---")
    print(f"Omega (Identical):     {omega_index(C1_over, C2_over):.4f}")
    print(f"NMI (Identical):       {overlapping_nmi(C1_over, C2_over):.4f}")
    print(f"F1 (Identical):        {overlapping_f1_score(C1_over, C2_over):.4f}")
    print(f"Jaccard (Identical):   {overlapping_jaccard_index(C1_over, C2_over):.4f}")
    print("---")
    print(f"Omega (Finer):         {omega_index(C1_over, C3_over):.4f}")
    print(f"NMI (Finer):           {overlapping_nmi(C1_over, C3_over):.4f}")
    print(f"F1 (Finer):            {overlapping_f1_score(C1_over, C3_over):.4f}")
    print(f"Jaccard (Finer):       {overlapping_jaccard_index(C1_over, C3_over):.4f}")
    print("---")
    print(f"Omega (Coarser):       {omega_index(C1_over, C4_over):.4f}")
    print(f"NMI (Coarser):         {overlapping_nmi(C1_over, C4_over):.4f}")
    print(f"F1 (Coarser):          {overlapping_f1_score(C1_over, C4_over):.4f}")
    print(f"Jaccard (Coarser):     {overlapping_jaccard_index(C1_over, C4_over):.4f}")
    print("---")
    print(f"Omega (Missing):       {omega_index(C1_over, C5_over):.4f}")
    print(f"NMI (Missing):         {overlapping_nmi(C1_over, C5_over):.4f}")
    print(f"F1 (Missing):          {overlapping_f1_score(C1_over, C5_over):.4f}")
    print(f"Jaccard (Missing):     {overlapping_jaccard_index(C1_over, C5_over):.4f}")