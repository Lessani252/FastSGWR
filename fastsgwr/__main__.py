import os
import click
from sgwr import fastsgwr
import shlex
import platform

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
    #command = f'mpiexec -np {np} python "{mpi_path}" -data "{data}" -out "{out}" -c'
    def safe_quote(path):
        if platform.system() == "Windows":
            # Use double quotes for Windows (cmd-compatible)
            return f'"{path}"'
        else:
            # Use shlex.quote for Unix/Linux/macOS
            return shlex.quote(path)

    mpi_path_escaped = safe_quote(mpi_path)
    data_escaped = safe_quote(data)
    out_escaped = safe_quote(out)

    command = f'mpiexec -np {np} python {mpi_path_escaped} -data {data_escaped} -out {out_escaped} -c'

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

