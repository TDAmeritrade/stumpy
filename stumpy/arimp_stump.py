# naive
def arimp_naive(T_A, m, exclusion_zone=None, row_wise=False):
    """
    Traverse distance matrix diagonally and update the matrix profile and
    matrix profile indices if the parameter `row_wise` is set to `False`.
    If the parameter `row_wise` is set to `True`, it is a row-wise traversal.
    """

    distance_matrix = np.array(
        [distance_profile(Q, T_A, m) for Q in core.rolling_window(T_A, m)]
    )
    T_B = T_A.copy()

    distance_matrix[np.isnan(distance_matrix)] = np.inf

    n_A = T_A.shape[0]
    n_B = T_B.shape[0]
    l = n_A - m + 1
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    SL = list([np.inf] for _ in range(l))
    SLI = list([-1] for _ in range(l))

    SR = list([np.inf] for _ in range(l))
    ISR = list([-1] for _ in range(l))

    RL = list([np.inf] for _ in range(l))
    RLI = list([-1] for _ in range(l))

    LR = list([np.inf] for _ in range(l))
    LRI = list([-1] for _ in range(l))

    if row_wise:
        for i in range(l):
            apply_exclusion_zone(distance_matrix[i], i, exclusion_zone, np.inf)

        for i, D in enumerate(distance_matrix):
            # self-join / AB-join: matrix proifle and indices
            idx = np.argmin(D)
            P[i, 0] = D[idx]
            if P[i, 0] == np.inf:
                idx = -1
            I[i, 0] = idx

            # self-join: left matrix profile
            if ignore_trivial and i > 0:
                IL = np.argmin(D[:i])
                if D[IL] == np.inf:
                    IL = -1
                I[i, 1] = IL

            # self-join: right matrix profile
            if ignore_trivial and i < D.shape[0]:
                IR = i + np.argmin(D[i:])  # shift argmin by `i` to get true index
                if D[IR] == np.inf:
                    IR = -1
                I[i, 2] = IR

    else:  # diagonal traversal
        if ignore_trivial:
            diags = np.arange(exclusion_zone + 1, n_A - m + 1)
        else:
            diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)

        for k in diags:
            if k >= 0:
                iter_range = range(0, min(n_A - m + 1, n_B - m + 1 - k))
            else:
                iter_range = range(-k, min(n_A - m + 1, n_B - m + 1 - k))

            for i in iter_range:
                D = distance_matrix[i, i + k]
                if D < P[i, 0]:
                    P[i, 0] = D
                    I[i, 0] = i + k

                if ignore_trivial:  # Self-joins only
                    if D < P[i + k, 0]:
                        P[i + k, 0] = D
                        I[i + k, 0] = i

                    if i < i + k:
                        # Left matrix profile and left matrix profile index
                        if D < P[i + k, 1]:
                            P[i + k, 1] = D
                            I[i + k, 1] = i

                        if D < P[i, 2]:
                            # right matrix profile and right matrix profile index
                            P[i, 2] = D
                            I[i, 2] = i + k

    result = np.empty((l, 4), dtype=object)
    result[:, 0] = P[:, 0]
    result[:, 1:4] = I[:, :]

    return result
