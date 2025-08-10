"""
"""
import sys

import json
import __main__
import subprocess
import pickle
import time
import os
from math import ceil
import numpy as np
import sys
import click
from slirm.slim import SlimRuns, read_params
from slirm.samplers import Sampler, ParamGrid
from slirm.utils import make_dirs

CMD = "squeue --user {user} -r --array-unique -h -t  pending,running --format='%.18i %.30j'"

JOBNAME = "slurm_slim"

DATADIR = '../../data/slim_sims/'

TEMPLATE = f"""\
#!/bin/bash
#SBATCH --chdir={{cwd}}
#SBATCH --error=logs/error/{JOBNAME}_%j.err
#SBATCH --output=logs/out/{JOBNAME}_%j.out
#SBATCH --account={{account}}
#SBATCH --partition={{partition}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu={{mem_per_cpu}}
#SBATCH --job-name={JOBNAME}_%j
#SBATCH --time={{job_time}}
#SBATCH --exclude=rome060

{{cmd}}

"""

def est_time(secs_per_job, batch_size, factor=5):
    tot_secs = secs_per_job * batch_size * factor
    tot_hours = tot_secs / 60 / 60
    days = int(tot_hours // 24)
    time_left = tot_hours % 24
    hours = int(time_left // 1)
    minutes = ceil(60*(time_left-hours))
    return f"{days:02d}-{hours}:{minutes:02d}:00"

def get_files(dir, suffix):
    results = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(suffix):
                results.append(os.path.join(root, file))
    return results


def query_jobs(user):
    "Return lines from squeue containing this jobname."
    run = subprocess.run(CMD.format(user=user), shell=True, capture_output=True)
    stdout = [x.strip() for x in run.stdout.decode().split('\n') if len(x)]
    # note: only look for part of jobname, squeue truncates
    jobs = [x.split()[0] for x in stdout if JOBNAME[:30] in x]
    return jobs

def make_job_script_lines(batch):
    """
    Write a command strig that concatenates all the lines for a job,
    and make the output directories if they don't exist.
    """
    rows = []
    dirs = []
    for job in batch:
        outfile, cmd = job
        assert not os.path.exists(outfile), f"file {outfile} exists!"
        dirs.append(os.path.split(outfile)[0])
        rows.append(cmd)
    unique_dirs = set(dirs)
    if not len(rows):
        return ""
    mkdirs = "mkdir -p " + " ".join(unique_dirs) + "\n"
    return mkdirs + "\n".join(rows) + "\n"

def make_job(batch, job_time, account, partition, mem_per_cpu):
    "Take a batch of (outfile, cmd) tuples and make a sbatch script to run the commands"
    cmd = make_job_script_lines(batch)
    sbatch = TEMPLATE.format(
        job_time=job_time,
        cwd=os.getcwd(),
        cmd=cmd,
        account=account,
        partition=partition,
        mem_per_cpu=mem_per_cpu
    )
    return sbatch



def job_dispatcher(user, jobs, max_jobs, batch_size, secs_per_job, account, partition, mem_per_cpu, sleep=30):
    """
    Submit multiple sbatch scripts through standard in.
    """
    t0 = time.time()
    nbatches = len(jobs)

    live_jobs = dict()
    done_jobs = dict()
    total_time = 0
    total_done = 0

    def update_jobs():
        running_jobs = query_jobs(user)
        nonlocal total_time
        nonlocal total_done
        for job in list(live_jobs):
            if job not in running_jobs:
                t1 = time.time()
                tdelta = t1 - live_jobs.pop(job)
                done_jobs[job] = (t1, tdelta)
                total_time += tdelta
                total_done += 1
        return running_jobs
    est_time_per_batch = est_time(secs_per_job, batch_size)
    sys.stderr.write(f"Estimated time per batch: {est_time_per_batch}\n")
    sys.stderr.flush()
    while True:
        running_jobs = update_jobs()
        while len(running_jobs) < max_jobs:
            # submit batches until the queue is full
            try:
                this_batch = jobs.pop()
            except IndexError:
                break
            sbatch_cmd = make_job(this_batch, job_time=est_time_per_batch, account=account, partition=partition, mem_per_cpu=mem_per_cpu)
            sys.stdout.write(sbatch_cmd)
            sys.stdout.flush()
            res = subprocess.run(["sbatch"], input=sbatch_cmd, text=True, capture_output=True)
            assert res.returncode == 0, f"sbatch had a non-zero ({res.returncode}) exit code, input args were:\n {res.args} !"
            jobid = res.stdout.strip().replace('Submitted batch job ', '')
            # put this new job in the tracker
            live_jobs[jobid] = time.time()

            running_jobs = update_jobs()
            # clean out the live jobs
            njobs = len(running_jobs)

            ave = total_time/total_done if total_done > 0 else 0
            line = f"{nbatches - len(jobs)}/{nbatches} ({100*np.round((nbatches - len(jobs))/nbatches, 3)}%) batches submitted, {njobs} jobs currently running, {len(done_jobs)} batches done, ~{np.round(ave/60, 2)} mins per job...\r"
            sys.stderr.write(line)
            sys.stderr.flush()

        if not len(jobs) and not len(live_jobs):
            break
        time.sleep(sleep)
    return time.time() - t0, done_jobs



@click.command()
@click.argument('config', type=click.File('r'), required=True)
@click.option('--user', required=True, type=str, help="your username (for job querying)")
@click.option('--dir', required=True, type=str, help="output directory for simulation results")
@click.option('--seed-dir', default=None,type=str, help="seed simulations directory")
@click.option('--suffix', required=True, type=str, help="suffix of output files (e.g. 'treeseq.tree')")
@click.option('--secs-per-job', required=True, type=int, help="number of seconds per simulation")
@click.option('--max-jobs', default=5000, show_default=True, help="max number of jobs before launching more")
@click.option('--seed', required=True, type=int, help='seed to use')
@click.option('--add-name', is_flag=True, help='add the name of the config file to the output file basenames')
@click.option('--no-calc', is_flag=True, help='run simulations without performing measurements')
@click.option('--split-dirs', default=3, show_default=True, type=int, help="number of seed digits to use as subdirectory")
@click.option('--slim', default='slim', show_default=True, help='path to SLiM executable')
@click.option('--max-array', default=None, show_default=True, type=int, help='max number of array jobs')
@click.option('--batch-size', default=None, show_default=True, type=int, help='size of number of sims to run in one job')
@click.option('--account', required=True, type=str, help='SLURM account name')
@click.option('--partition', required=True, type=str, help='SLURM partition')
@click.option('--mem-per-cpu', default='4G', show_default=True, type=str, help='Memory per CPU (e.g., 4G)')
def generate(config, user, dir, seed_dir, suffix, secs_per_job, max_jobs, seed, split_dirs,
             slim, max_array, batch_size, add_name, no_calc, account, partition, mem_per_cpu):
    config = json.load(config)

    # note: we package all the sim seed-based subdirs into a sims/ directory
    if config['runtype'] == 'grid':
        sampler = ParamGrid
    else:
        assert config['runtype'] == 'samples', "config file must have 'grid' or 'samples' runtype."
        sampler = Sampler

    run = SlimRuns(config, dir=dir, seed_dir=seed_dir, sims_subdir=True, sampler=sampler, add_name=add_name,
                   split_dirs=split_dirs if split_dirs>0 else None, seed=seed, nocalc=no_calc)

    # get the existing files
    print("searching for existing simulation results...   ", end='')
    existing = get_files(run.dir, suffix)  # run.dir has the name included
    print("done.")
    print(f"{len(existing):,} result files have been found -- these are ignored.")

    # generate and batch all the sims
    run.generate(suffix=suffix, ignore_files=existing, package_basename=True, package_rep=True)
    total_size = len(run.runs)
    #import pdb;pdb.set_trace()
    if not total_size:
        print("no files need to be generated, exiting successfully")
        sys.exit(0)
    print(f"beginning dispatching of {total_size:,} simulations...")
    job_batches = run.batch_runs(batch_size=batch_size, slim_cmd=slim)

    # turn these into a list
    job_batches = list(job_batches.values())

    # set out the output directory
    sim_dir = make_dirs(dir, config['name'])
    total_time, done_jobs = job_dispatcher(user, job_batches, max_jobs, batch_size, secs_per_job, account, partition, mem_per_cpu)
    print(f"\n\ntotal run time: {str(total_time)}")
    with open(f"{config['name']}_stats.pkl", 'wb') as f:
        pickle.dump(done_jobs, f)

def main():
    print("Starting slirm...")
    print(f"__name__ is {__name__}")
    
if __name__ == "slirm.cli":
    generate()


