"""
Originally, the code was developed based on FastGWR, authored by Ziqi Li (liziqi1992@gmail.com).
It has been revised to accommodate the SGWR model by the author, M. Naser Lessani.
"""

import argparse
from mpi4py import MPI
from FastSGWR import SGWR

if __name__ == "__main__":
    # Initializing MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-data")
    parser.add_argument("-out", default="model_results.csv")
    parser.add_argument("-bw")
    parser.add_argument("-minbw")
    parser.add_argument("-chunks", default=1)
    parser.add_argument("-standardize", action='store_true')  # NEW
    parser.add_argument('-estonly', action='store_true')
    parser.add_argument('-c', '--constant', action='store_true')
    parser.add_argument('-a', '--adaptive', action='store_true')
    parser.add_argument('-f', '--fixed', action='store_true')
    parser.add_argument('-ac', '--alphacurve', action='store_true')
    parser.add_argument('-gwr', '--gwr', action='store_true')
    parser.add_argument('-biga', '--biga', action='store_true')

    n_chunks = parser.parse_args().chunks

    # Fitting the model
    SGWR(comm, parser).fit()





