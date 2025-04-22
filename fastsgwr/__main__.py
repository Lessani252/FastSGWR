import os
import click
# import fastsgwr
from sgwr import fastsgwr

@click.group()
@click.version_option("0.2.9")
def main():
    pass

@main.command()
@click.option("-np", required=True, type=int)
@click.option("-data", required=True)
@click.option("-out", default="model_results.csv")
@click.option("-bw", default=None)
@click.option("-minbw", default=None)
@click.option("-chunks", default=None)
@click.option("-estonly", is_flag=True)
@click.option("-adaptive/-fixed", default=True)
@click.option("-standardize", is_flag=True, default=False)
@click.option("-alphacurve", is_flag=True, default=False)
@click.option("-gwr", is_flag=True, default=False)
@click.option("-biga", is_flag=True, default=False)

def run(np, data, out, adaptive, bw, minbw, chunks, estonly, standardize, alphacurve, gwr, biga):
    mpi_path = os.path.join(os.path.dirname(fastsgwr.__file__), 'sgwr_mpi.py')
    output = os.path.dirname(data)
    out = os.path.join(output, out)

    # Build the mpi command
    command = f'mpiexec -np {np} python "{mpi_path}" -data "{data}" -out "{out}" -c'

    if bw is not None and bw != "":
        command += f' -bw {bw}'
    if minbw is not None and minbw != "":
        command += f' -minbw {minbw}'
    if chunks is not None and chunks != "":
        command += f' -chunks {chunks}'
    if estonly:
        command += ' -estonly'
    if alphacurve:
        command += ' -ac'
    if standardize:
        command += ' -standardize'
    if gwr:
        command += ' -gwr'
    if biga:
        command += ' -biga'
    # and -f for fixed.
    command += ' -a' if adaptive else ' -f'

    os.system(command)

if __name__ == "__main__":
    main()

