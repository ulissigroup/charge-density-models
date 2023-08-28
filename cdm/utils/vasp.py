import os
import yaml
import re
import torch

import ase
from ase.calculators.vasp import Vasp

from tqdm.notebook import tqdm

from cdm.utils.inference import inference
from cdm.utils.preprocessing import VaspChargeDensity

def write_CHGCAR_like(
    model,
    input_CHGCAR_path,
    output_CHGCAR_path,
    batch_size = 100000,
    use_tqdm = False,
    device = 'cuda',
):      
    vcd = VaspChargeDensity(input_CHGCAR_path)
    vcd.chg[0] = inference(
        atoms = vcd.atoms[0],
        model = model,
        grid = vcd.chg[0].shape,
        atom_cutoff = model.atom_message_model.cutoff,
        probe_cutoff = model.probe_message_model.cutoff,
        batch_size = batch_size,
        use_tqdm = use_tqdm,
        device = device,
        total_density = torch.sum(torch.tensor(vcd.chg[0])),
    )
    vcd.write(output_CHGCAR_path)
    
def setup_VASP_experiment(
    input_augs_path,
    output_path,
    input_density_path = None,
    model = None,
    batch_size = 100000,
    device = 'cuda',
):
    paths = os.listdir(input_augs_path)
    assert set(paths) == set(os.listdir(output_path))
    
    if model is None:
        assert input_density_path is not None
        assert set(paths) == set(os.listdir(input_density_path))
        
        for i in tqdm(paths):
            vcd = VaspChargeDensity(os.path.join(input_density_path, i, 'CHGCAR'))
            vcd.aug_dict = VaspChargeDensity(os.path.join(input_augs_path, i, 'CHGCAR')).aug_dict
            vcd.write(os.path.join(output_path, i, 'CHGCAR'))
        
    elif (input_density_path is None) or (input_density_path == input_augs_path):
        assert model is not None
        
        for i in tqdm(paths):
            write_CHGCAR_like(
                model = model,
                input_CHGCAR_path = os.path.join(input_augs_path, i, 'CHGCAR'),
                output_CHGCAR_path = os.path.join(output_path, i, 'CHGCAR'),
                batch_size = batch_size,
                device = device,
                use_tqdm = True,
            )
    else:
            assert set(paths) == set(os.listdir(input_density_path))
            
            for i in tqdm(paths):
                
                vcd = VaspChargeDensity(os.path.join(input_density_path, i, 'CHGCAR'))
                vcd.chg[0] = inference(
                    atoms = vcd.atoms[0],
                    model = model,
                    grid = vcd.chg[0].shape,
                    atom_cutoff = model.atom_message_model.cutoff,
                    probe_cutoff = modle.probe_message_model.cutoff,
                    batch_size = batch_size,
                    use_tqdm = False,
                    device = device,
                    total_density = torch.sum(torch.tensor(vcd.chg[0])),
                )
                vcd.aug_dict = VaspChargeDensity(os.path.join(input_augs_path, i, 'CHGCAR')).aug_dict
                vcd.write(os.path.join(output_path, i, 'CHGCAR'))
        
def kubernetes_VASP_batch(
    path,
    base_params,
    exp_tag,
    namespace,
    cpu_req = 4,
    cpu_lim = 16,
    mem_req = '8Gi',
    mem_lim = '16Gi',
    VASP_mp_threads = 16,
    select_only = None,
):
    '''
    args:
        path (str): the directory that contains your VASP directories
        base_params: a yaml of parameters obtained from your template file
                     or a path to the template file
        exp_tag: a string to identify this experiment/run/batch
        cpu_req: how many CPU cores to request for each individual VASP job
        cpu_lim: how many CPU cores to limit the job to (if extra cores are available on the node)
        mem_req: how much memory to reserve for each individual VASP job
        mem_lim: how much memory to limit the job to (if extra memory is avaialable on the node)
        VASP_mp_threads: how many threads to launch for your VASP calculation.
                    Should be between (or equal to) cpu_req and cpu_lim
        
    This script is designed to help submit calculations to Laikapack, a
    Kubernetes cluster at CMU. It is likely that your HPC setup is
    different. In that case, you should use whatever submission system is 
    recommended for VASP calculations on your resources.
    
    If None is passed in for any of the resource values, this script will not attempt to replace that
    value in the template file. In this case, your template file should specify the resource amount.
        
    '''
    
    if select_only is not None:
        paths = select_only
    else:
        paths = os.listdir(path)
        
    if isinstance(base_params, str):
        with open(base_params, 'r') as stream:
            base_params = yaml.safe_load(stream)
        
    params = base_params.copy()
    params['metadata']['namespace'] = namespace
    params['spec']['template']['spec']['containers'] = [params['spec']['template']['spec']['containers']]
    container = params['spec']['template']['spec']['containers'][0]
    
    # Set the resource usage, which is assumed to be the same for all jobs
    if cpu_lim is not None:
        container['resources']['limits']['cpu'] = cpu_lim
    if cpu_req is not None:
        container['resources']['requests']['cpu'] = cpu_req
    if VASP_mp_threads is not None:
        container['args'][0] = re.sub('-np \d+', '-np '+str(VASP_mp_threads), container['args'][0])
    
    if mem_lim is not None:
        container['resources']['limits']['memory'] = mem_lim
    if mem_req is not None:
        container['resources']['requests']['memory'] = mem_req
    
    for fid, directory in enumerate(tqdm(paths)):
        # Set the metadata for each job
        params['metadata']['name'] = exp_tag + '-' + re.sub('_', '-', directory)
        container['workingDir'] = path + '/' + directory
        container['name'] = exp_tag + '-' + re.sub('_', '-', directory)
        
        # Write the job specification file
        with open('job.yaml', 'w') as config_file:
            yaml.dump(params, config_file, default_flow_style = False)
            
        # Submit the job    
        os.system('kubectl apply -f job.yaml > /dev/null')
        
def read_experiment(
    path,
    loud = False,
    skip = None,
):
    paths = os.listdir(path)
    
    if skip is not None:
        for i in skip:
            paths.remove(i)
    
    results = {}
    
    for fid, directory in enumerate(tqdm(paths)):
        if loud:
            print(directory)
            
        if directory not in []:
            
            try:
                with open(path + '/' + directory + '/' + 'vasp.out') as vaspout:
                    
                    ncg = 0
                    DAVS = [line for line in vaspout if ('DAV' in line)]
                    for idx, line in enumerate(DAVS):
                        line = line.split(' ')
                        line = [string for string in line if (string != '')]
                        DAVS[idx] = line
                        ncg += int(line[5])

                    n = len(DAVS)
                    E = float(DAVS[-1][2])

                    results[directory] = {'num_scf_steps': n, 'Energy': E, 'ncg': ncg}
            except Exception as exception:
                print(f'Error: could not read vasp.out from: {path}/{directory}')
                print(str(exception))
            
    return results

def compare_VASP_experiments(
    baseline,
    trial,
    skip = None,
):
    baseline = read_experiment(baseline, skip = skip)
    trial = read_experiment(trial, skip = skip)
    
    assert set(baseline.keys()) == set(trial.keys())
    
    E_diffs = []
    for i in baseline.keys():
        E_diffs.append(trial[i]['Energy'] - baseline[i]['Energy'])
        ncg_fraction = sum(x['ncg'] for x in trial.values()) / sum(x['ncg'] for x in baseline.values())
        indiv_ncg_fractions = [x['ncg'] / y['ncg'] for x, y in zip(trial.values(), baseline.values())]
        
        faster_fraction = len([x for x in trial.keys() if trial[x]['ncg'] < baseline[x]['ncg']]) / len(trial.keys())
        slower_fraction = len([x for x in trial.keys() if trial[x]['ncg'] > baseline[x]['ncg']]) / len(trial.keys())
        equal_fraction = len([x for x in trial.keys() if trial[x]['ncg'] == baseline[x]['ncg']]) / len(trial.keys())
        
    return {
        'faster_fraction': faster_fraction,
        'slower_fraction': slower_fraction,
        'equal_fraction': equal_fraction,
        'ncg_fraction': ncg_fraction,
        'E_diffs': E_diffs,
        'indiv_ncg_fractions': indiv_ncg_fractions,
    }