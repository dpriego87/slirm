# slim.py -- helpers for snakemake/slim sims

import os
import copy
from math import ceil
from collections import defaultdict
import itertools
import warnings
import numpy as np
from slirm.utils import signif
from slirm.sim_utils import param_grid, read_params, random_seed

def filename_pattern(dir, base, params, split_dirs=False, seed=False, rep=False):
    """
    Create a filename pattern with wildcares in braces (for Snakemake)
    from a basename 'base' and list of parameters. If 'seed' or 'rep' are
    True, these will be added in manually.

    Example:
      input: base='run', params=['h', 's']
      output: 'run_h{h}_s{s}'
    """
    param_str = [v + '{' + v + '}' for v in params]
    if seed:
        param_str.append('seed{seed}')
    if rep:
        param_str.append('rep{rep}')
    if split_dirs:
        base = os.path.join(dir, '{subdir}', base)
    else:
        base = os.path.join(dir, base)
    pattern = base + '_'.join(param_str)
    return pattern


def slim_call(param_types, script, slim_cmd="slim", add_seed=False,
              add_rep=False, manual=None):
    """
    Create a SLiM call prototype for Snakemake, which fills in the
    wildcards based on the provided parameter names and types (as a dict).

    param_types: dict of param_name->type entries
    slim_cmd: path to SLiM
    seed: bool whether to pass in the seed with '-s <seed>' and add a
            seed between 0 and 2^63
    manual: a dict of manual items to pass in
    """
    call_args = []
    for p, val_type in param_types.items():
        is_str = val_type is str
        if is_str:
            # silly escapes...
            val = f'\\"{{wildcards.{p}}}\\"'
        else:
            val = f"{{wildcards.{p}}}"
        call_args.append(f"-d {p}={val}")
    add_on = ''
    if manual is not None:
        # manual stuff
        add_on = []
        for key, val in manual.items():
            if isinstance(val, str):
                add_on.append(f'-d {key}=\\"{val}\\"')
            else:
                add_on.append(f'-d {key}={val}')
        add_on = ' ' + ' '.join(add_on)
    if add_seed:
        call_args.append("-s {wildcards.seed}")
    if add_rep:
        call_args.append("-d rep={wildcards.rep}")
    full_call = f"{slim_cmd} " + " ".join(call_args) + add_on + " " + script
    return full_call


class SlimRuns(object):
    def __init__(self, config, dir='.', sims_subdir=False, sampler=None,
                 split_dirs=None, seed=None):
        msg = "runtype must be 'grid' or 'samples'"
        assert config.get('runtype', None) in ['grid', 'samples'], msg
        self.runtype = config['runtype']
        self.name = config['name']
        self.nreps = config.get('nreps', None)
        if self.is_samples:
            # this is the number of total samples *not* including replicates,
            # e.g. unique parameter combinations
            self.nsamples = config['nsamples']
        else:
            self.nsamples = None # figured out from grid sizes, etc

        self.script = config['slim']
        msg = f"SLiM file '{self.script}' does not exist"
        assert os.path.exists(self.script), msg

        self.params, self.param_types = read_params(config)
        self.add_seed = True
        if split_dirs is not None:
            # this is to prevent thousands/millions of simulation files going
            # to same directory
            assert isinstance(split_dirs, int), "split_dirs needs to be int"
            # we need to pass in the subdir
            self.param_types = {'subdir': str, **self.param_types}
        if sims_subdir:
            self.dir = os.path.join(dir, self.name, 'sims')
        else:
            self.dir = os.path.join(dir, self.name)
        self.split_dirs = split_dirs
        self.basename = f"{self.name}_"
        self.seed = seed if seed is not None else random_seed()
        self.sampler_func = sampler
        if sampler is None and self.is_samples:
            raise ValueError("no sampler function specified and runtype='samples'")
        # for when we instantiate the sampler with seeds, etc
        # sampler also can be grid (non-random)
        self.sampler = None
        self.batches = None

    def _generate_runs(self, suffix, ignore_files=None, package_basename=True, package_rep=True):
        """
        For samplers only, not param grids!

        ignore_files is an option set of files to exclude, e.g. stuff that's
        already been simulated.

        package_rep is a bool indicating whether to package in the replicate
        number 'rep' into the sample's dictionary.
        """
        suffix_is_str = False
        if isinstance(suffix, str):
            suffix_is_str = True
            suffix = [suffix]

        ignore_files = set() if ignore_files is None else set([os.path.basename(f) for f in ignore_files])
        targets = []
        runs = []
        nreps = 1 if self.nreps is None else self.nreps
        ignored = set()
        for sample in self.sampler:
            # each sample is a dict of params that are like Snakemake's
            # wildcards
            for rep in range(nreps):
                # draw nreps samples
                sample = copy.copy(sample)
                # we have more than one replicate, so we need to use the same
                # params, but with a different random seed
                seed = random_seed(self.sampler.rng)
                sample['seed'] = seed

                if self.nreps is not None or package_rep:
                    # package_rep is whether to include 'rep' into sample dict
                    sample['rep'] = rep
                # check if we need to add in a subdir:
                if self.split_dirs is not None:
                    dir_seed = str(sample['seed'])[:self.split_dirs]
                    sample = {**sample, 'subdir': dir_seed}
                if package_basename:
                    # this is a sim basename, which has all the parameters
                    # slightly different than the basename in this class.
                    basename = os.path.basename(
                        self.filename_pattern.format(**sample))
                    sample = {**sample, 'basename': basename}
                # this is the parent dir of subdir (if set) or wherever simulations are being saved
                # to later be passed in the slim command line
                sample = {**sample, 'dir': self.dir}
                # the expected output files for this run
                target_files = []
                for end in suffix:
                    filename = f"{self.filename_pattern}_{end}"
                    # propagate the sample into the filename
                    filename = filename.format(**sample)
                    target_files.append(filename)

                needed_target_files = [os.path.basename(f) not in ignore_files for f in target_files]
                if any(needed_target_files):
                    # if we only have one target file per sim, just add that not in a list
                    if suffix_is_str:
                        assert len(target_files) == 1
                        targets.append(target_files[0])
                    else:
                        targets.append(target_files)
                    runs.append(sample)
                else:
                    assert suffix_is_str
                    ignored.add(target_files[0])

        # find weird files -- not in target, not in ignored
        ignored = set(os.path.basename(f) for f in ignored)
        # certain files are not being ignored even though they exist
        weird = ignore_files.difference(ignored)
        if len(weird) > 0:
            warnings.warn("there are existing simulation results that are not being ignored!")
        #target_set = set(os.path.basename(f) for f in targets)
        #not_in_targets = [f for f in weird if f not in target_set]
        #import pdb;pdb.set_trace()
        self.targets = targets
        self.runs = runs
        assert len(self.targets) == len(self.runs)

    def generate(self, suffix, ignore_files=None, package_basename=True, package_rep=True):
        """
        Run the sampler to generate samples or expand out the parameter grid.
        """
        if self.is_grid:
            self.sampler = self.sampler_func(self.params, add_seed=True, seed=self.seed)
        else:
            self.sampler = self.sampler_func(self.params, total=self.nsamples,
                                             add_seed=True, seed=self.seed)
        self._generate_runs(suffix=suffix, ignore_files=ignore_files,
                            package_basename=package_basename,
                            package_rep=package_rep)


    @property
    def total_draws(self):
        "SlimRuns.nsamples x SlimRuns.nreps"
        return self.nsamples if not self.has_reps else self.nsamples*self.nreps

    def batch_runs(self, batch_size=1, slim_cmd='slim'):
        """
        Create a dictionary of array index (e.g. from Slurm) --> list of
        sample indices. This is a 1-to-1 mapping if batch_size = 1, otherwise

        """
        assert self.runs is not None, "runs not generated!"
        n = len(self.runs)
        assert n > 0

        # get cmds
        runs = self.runs
        nruns = len(self.runs)
        groups = np.split(np.arange(nruns), np.arange(0, nruns, batch_size)[1:])
        self.batches = {i: grp for i, grp in enumerate(groups)}

        self.job_batches = defaultdict(list)
        for idx in self.batches:
            for job_idx in self.batches[idx]:
                wildcards = self.runs[job_idx]
                file = self.targets[job_idx]
                cmd = self.slim_command(wildcards, slim_cmd=slim_cmd)
                job = (file, cmd)
                self.job_batches[idx].append(job)
        return self.job_batches

    @property
    def has_reps(self):
        return self.nreps is not None or self.nreps > 1

    @property
    def is_grid(self):
        return self.runtype == 'grid'

    @property
    def is_samples(self):
        return self.runtype == 'samples'

    def slim_call(self, slim_cmd='slim', manual=None):
        """
        Return a string SLiM call, with SLiM variables passed as command
        line arguments and retrieved from SLiM wildcards, e.g.
        slim -d val={wildcards.val} (for use with a structured filename).

        Passes in the name of the run.
        """
        name = {'name': self.name}
        if manual is not None:
            manual = {**name, **manual}
        else:
            manual = name

        return slim_call(self.param_types, self.script, slim_cmd=slim_cmd,
                         add_seed=self.add_seed, add_rep=self.has_reps,
                         manual=manual)

    def slim_command(self, wildcards, **slim_call_kwargs):
        if 'basename' in wildcards:
            # this is not a parameter, so we include it in manual
            manual = slim_call_kwargs.get('manual', {})
            manual['basename'] = wildcards.pop('basename')
            slim_call_kwargs['manual'] = manual
        if 'dir' in wildcards:
            # this is not a parameter, so we include it in manual
            manual = slim_call_kwargs.get('manual', {})
            manual['dir'] = wildcards.pop('dir')
            slim_call_kwargs['manual'] = manual
        call = self.slim_call(**slim_call_kwargs).replace("wildcards.", "")
        return call.format(**wildcards)


    def slim_commands(self, **slim_call_kwargs):
        call = self.slim_call(**slim_call_kwargs).replace("wildcards.", "")
        if self.runs is None:
            raise ValueError("run SlimRuns.generate()")
        for wildcards in self.runs:
            yield call.format(**wildcards)

    @property
    def filename_pattern(self):
        """
        Return the filename pattern with wildcards.
        """
        return filename_pattern(self.dir, self.basename, self.params.keys(),
                                split_dirs=self.split_dirs is not None,
                                seed=self.add_seed, rep=self.has_reps)

    def wildcard_output(self, suffix):
        """
        For the 'output' entry of a Snakemake rule, this returns the
        expected output files with wildcards, with suffix attached.
        'suffix' can be a list/tuple of many outputs or a string.
        """
        if isinstance(suffix, (list, tuple)):
            return [f"{self.filename_pattern}_{end}" for end in suffix]
        elif isinstance(suffix, str):
            return f"{self.filename_pattern}_{suffix}"
        else:
            raise ValueError("suffix must be list/tuple or str")

    @property
    def param_order(self):
        """
        SLiM constructs filename automatically too; this is the order, as a
        string of parameters, to use for the filename_str() function.
        """
        return ', '.join(f"'{v}'" for v in self.params.keys())

