"""A cubic exact Pfaffian elimination.

The routine uses skew Gaussian elimination.  It is deliberately small and
has no numerical fallback: entries may be SymPy expressions or any exact
field elements supporting the four arithmetic operations.
"""

from __future__ import annotations


def pfaffian(matrix):
    """Return the Pfaffian of an even skew matrix in ``O(size**3)``.

    ``matrix`` can be a SymPy matrix or a nested sequence.  Pivoting is by
    simultaneous row/column permutation, so every swap has the correct
    Pfaffian sign.  No square root of a determinant is taken.
    """

    rows = [list(row) for row in matrix.tolist()] if hasattr(matrix, "tolist") else [list(row) for row in matrix]
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("the Pfaffian requires a square matrix")
    if size % 2:
        raise ValueError("the Pfaffian requires even size")
    if size == 0:
        return 1

    answer = 1
    for first in range(0, size, 2):
        second = first + 1
        pivot_column = next(
            (column for column in range(second, size) if rows[first][column] != 0),
            None,
        )
        if pivot_column is None:
            return 0
        if pivot_column != second:
            rows[second], rows[pivot_column] = rows[pivot_column], rows[second]
            for row in rows:
                row[second], row[pivot_column] = row[pivot_column], row[second]
            answer = -answer

        pivot = rows[first][second]
        answer *= pivot
        for row in range(second + 1, size):
            for column in range(row + 1, size):
                value = rows[row][column] - (
                    rows[first][row] * rows[second][column]
                    - rows[first][column] * rows[second][row]
                ) / pivot
                rows[row][column] = value
                rows[column][row] = -value
    return answer


def pfaffian_recursive(matrix):
    """Division-free Pfaffian expansion for small symbolic matrices.

    Gaussian Pfaffian elimination is cubic but introduces large transient
    rational denominators.  For matrices up to about size ten, the direct
    matching recurrence is often much faster over multivariate rational
    functions because it uses additions and multiplications only.
    """

    rows = (
        tuple(tuple(row) for row in matrix.tolist())
        if hasattr(matrix, "tolist")
        else tuple(tuple(row) for row in matrix)
    )
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("the Pfaffian requires a square matrix")
    if size % 2:
        raise ValueError("the Pfaffian requires even size")

    from functools import lru_cache

    @lru_cache(None)
    def visit(indices):
        if not indices:
            return 1
        first = indices[0]
        answer = 0
        for position in range(1, len(indices)):
            second = indices[position]
            remaining = indices[1:position] + indices[position + 1 :]
            answer += (-1) ** (position + 1) * rows[first][second] * visit(
                remaining
            )
        return answer

    return visit(tuple(range(size)))
