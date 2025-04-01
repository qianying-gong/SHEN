import csv
import fcntl
import time
import yaml
import os
from remote_eval.HardwareRemoteEvaluator import HardwareRemoteEvaluator


def acquire_lock(lock_file):
    lock = open(lock_file, 'w')
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except IOError:
            time.sleep(0.1)

def release_lock(lock):
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def get_mapping_file_path(model, dataflow):
    mapping_dict = {
        'ws': "data/mapping/" + model + "ee_ws.m",
        'os': "data/mapping/" + model + "ee_os.m",
        'rs': "data/mapping/" + model + "ee_rs.m"
    }

    if dataflow in mapping_dict:
        return mapping_dict[dataflow]
    else:
        raise ValueError(f"Unsupported dataflow: {dataflow}")


def save_population_to_csv(population, filename='population_results.csv'):
    if not population:
        print("Warning: Population is empty. CSV will only contain headers.")

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    file_path = os.path.join(script_dir, filename)


    fieldnames = ['position', 'dataflow', 'numPEs', 'L1Size', 'L2Size', 'error', 'latency', 'energy']

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        for individual in population:
            individual_dict = {
                'position': ','.join(map(str, individual.get('position'))),
                'dataflow': individual.get('dataflow'),
                'numPEs': individual.get('numPEs'),
                'L1Size': individual.get('L1Size'),
                'L2Size': individual.get('L2Size'),
                'error': individual.get('error'),
                'latency': individual.get('latency'),
                'energy': individual.get('energy')
            }
            writer.writerow(individual_dict)


def remote_eval_hardware(model, population, position, exit_layers, exit_counts, error):
    config_file_path = './config_evo.yml'
    config = load_config(config_file_path)

    remote_host = config['remote_evaluation']['remote_host']
    username = config['remote_evaluation']['username']
    key_file_path = config['remote_evaluation']['key_file_path']

    evaluator = HardwareRemoteEvaluator(remote_host, username, key_file_path)
    lock_file_path = "/tmp/docker_lockfile"

    for individual in population:

        dataflow = individual.get('dataflow')
        numPEs = individual.get('numPEs')
        L1Size = individual.get('L1Size')
        L2Size = individual.get('L2Size')

        hw_params = {
            "num_pes": numPEs,
            "l1_size": L1Size,
            "l2_size": L2Size,
            "noc_bw": 16,
            "noc_num_hops": 1
        }

        mapping_file_path = get_mapping_file_path(model, dataflow)

        lock = acquire_lock(lock_file_path)
        try:
            evaluator.execute_remote_command(hw_params, mapping_file_path)

            script_path = ""

            latency, energy = evaluator.execute_remote_csv_processing(model, dataflow, script_path, exit_layers, exit_counts)
        finally:
            release_lock(lock)

        individual['position'] = position
        individual['latency'] = latency
        individual['energy'] = energy
        individual['error'] = error

    save_population_to_csv(population)