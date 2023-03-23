import os
import yaml
import re

import ase
from ase.calculators.vasp import Vasp

from tqdm import tqdm

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
        atoms = vcd.atoms,
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
            )
    else:
            assert set(paths) == set(os.listdir(input_density_path))
            
            for i in tqdm(paths):
                vcd = VaspChargeDensity(os.path.join(input_density_path, i, 'CHGCAR'))
                vcd.chg[0] = inference(
                    atoms = vcd.atoms,
                    model = model,
                    grid = vcd.chg[0].shape,
                    atom_cutoff = model.atom_message_model.cutoff,
                    probe_cutoff = modle.probe_message_model.cutoff,
                    batch_size = batch_size,
                    use_tqdm = use_tqdm,
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
    cpu_cores = 4,
    memory = '8Gi',
    select_only = None,
):
    '''
    args:
        path (str): the directory that contains your VASP directories
        base_params: a yaml of parameters obtained from your template file
                     or a path to the template file
        exp_tag: a string to identify this experiment/run/batch
        cpu_cores: how many CPU cores for each individual VASP job
        memory: how much memory for each individual VASP job
        
    This script is designed to help submit calculations to Laikapack, a
    Kubernetes cluster at CMU. It is likely that your HPC setup is
    different. In that case, you should use whatever submission system is 
    recommended for VASP calculations on your resources.
        
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
    if cpu_cores is not None:
        container['resources']['limits']['cpu'] = cpu_cores
        container['resources']['requests']['cpu'] = cpu_cores
        container['args'][0] = re.sub('-np \d+', '-np '+str(cpu_cores), container['args'][0])
    
    if memory is not None:
        container['resources']['limits']['memory'] = memory
        container['resources']['requests']['memory'] = memory
    
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

                    DAVS = [line for line in vaspout if ('DAV' in line)]
                    for idx, line in enumerate(DAVS):
                        line = line.split(' ')
                        line = [string for string in line if (string != '')]
                        DAVS[idx] = line

                    n = len(DAVS)
                    E = float(DAVS[-1][2])

                    results[directory] = {'num_scf_steps':n, 'Energy': E}
            except:
                print(f'Error: could not read vasp.out from: {path}/{directory}')
            
    return results

def compare_VASP_experiments(
    baseline,
    trial,
    skip = None,
):
    baseline = read_experiment(baseline, skip = skip)
    trial = read_experiment(trial, skip = skip)
    
    assert set(baseline.keys()) == set(trial.keys())
    
    savings = []
    E_diffs = []
    for i in baseline.keys():
        savings.append(trial[i]['num_scf_steps'] / baseline[i]['num_scf_steps'])
        E_diffs.append(trial[i]['Energy'] - baseline[i]['Energy'])
    return savings, E_diffs