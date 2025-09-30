import os, multiprocessing as mp, os as _os
n = int(os.getenv("NUMPROC", "10"))
_os.cpu_count = lambda: n
mp.cpu_count = lambda: n

