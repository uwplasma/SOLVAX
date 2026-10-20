"""Worker process for the ``python-mumps`` backend, under an interpreter that matches its NumPy.

Started as ``python _mumps_worker.py`` (it imports nothing from solvax or JAX). Requests are pickled
``(command, payload)`` tuples of plain Python objects and raw array bytes, replies ``("ok", value)``
or ``("error", message)``.
"""

import os
import pickle
import sys
from typing import Any

import mumps
import numpy as np
import scipy.sparse as sparse

_INFOG = (1, 16, 21, 26, 27)


def _array(raw, dtype, shape):
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def main() -> None:
    requests, replies = sys.stdin.buffer, os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # MUMPS prints on unit 6; keep it off the reply channel
    context: Any = None
    while True:
        try:
            command, payload = pickle.load(requests)
        except EOFError:
            return
        try:
            if command == "analyze":
                rows, cols, data, dtype, shape, icntl = payload
                matrix = sparse.coo_array(
                    (_array(data, dtype, -1), (_array(rows, "i8", -1), _array(cols, "i8", -1))),
                    shape=shape,
                )
                context = mumps.Context()
                context.set_matrix(matrix)
            else:
                (icntl,) = payload[:1]
            instance = context.mumps_instance
            for index, value in icntl.items():
                if index != 22:  # python-mumps sets ICNTL(22) from ``ooc``
                    instance.icntl[index] = value
            if command == "analyze":
                context.analyze()
                value = None
            elif command == "factor":
                context.factor(reuse_analysis=True, ooc=bool(icntl.get(22)))
                value = None
            else:
                _, raw, dtype, shape = payload
                value = context.solve(_array(raw, dtype, shape)).tobytes()
            value = (value, {i: int(instance.infog[i]) for i in _INFOG})
            pickle.dump(("ok", value), replies)
        except BaseException as error:  # noqa: BLE001 - reported to the parent
            pickle.dump(("error", f"{type(error).__name__}: {error}"), replies)
        replies.flush()


if __name__ == "__main__":
    main()
