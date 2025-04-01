import torch
import numpy as np
import csv
import os
from EarlyExitNetwork import train_networks as tn, aux_funcs as af
from EarlyExitNetwork import early_exit_experiments as eee

def choose_indices(indices, count, is_ones):
    rng = np.random.default_rng()
    selected = []
    while len(selected) < count and indices:
        idx = indices.pop(0)
        if is_ones:
            indices = [x for x in indices if abs(x - idx) > 1]
        selected.append(idx)
    return selected


def fit_error(dataset, model, pos_exits):

    pos_exits = [pos_exits[i:i + 9] for i in range(0, len(pos_exits), 9)]
    af.set_random_seeds()
    device = af.get_pytorch_device()
    models_path = ''.format(af.get_random_seed())

    tn.train_models(dataset, model, models_path, pos_exits, device)

    torch.manual_seed(af.get_random_seed())  # reproducible
    np.random.seed(af.get_random_seed())

    trained_models_path = ''.format(af.get_random_seed())

    early_exit_counts, error = eee.early_exit_experiments(dataset, model, trained_models_path, pos_exits, device)

    return early_exit_counts, error


def adjust_initial_positions(positions, min_ones, max_ones):
    rng = np.random.default_rng()
    n, m = positions.shape

    for i in range(n):
        current_ones = np.sum(positions[i] == 1)
        if current_ones > max_ones:
            ones_indices = list(np.where(positions[i] == 1)[0])
            rng.shuffle(ones_indices)
            selected_indices = choose_indices(ones_indices, max_ones, is_ones=True)
            positions[i] = np.zeros(m, dtype=int)
            positions[i, selected_indices] = 1
        elif current_ones < min_ones:
            zeros_indices = list(np.where(positions[i] == 0)[0])
            rng.shuffle(zeros_indices)
            selected_indices = choose_indices(zeros_indices, min_ones, is_ones=False)
            positions[i, selected_indices] = 1

    return positions


def find_longest_zero_sequence(position_origin):
    max_start_idx = -1
    max_length = 0
    current_start_idx = -1
    current_length = 0
    for i in range(len(position_origin)):
        if position_origin[i] == 0:
            if current_length == 0:
                current_start_idx = i
            current_length += 1
        else:
            if current_length > max_length:
                max_start_idx = current_start_idx
                max_length = current_length
            current_length = 0

    if current_length > max_length:
        max_start_idx = current_start_idx
        max_length = current_length

    return max_start_idx, max_length


def adjust_single_position(unique_positions, global_best_position, position_origin, min_ones):
    rng = np.random.default_rng()

    def generate_new_position():
        diff = global_best_position ^ position_origin
        diff_indices = np.where(diff != 0)[0]
        diff_nums = len(diff_indices)
        if diff_nums == 0:
            return position_origin
        modified_diff = rng.integers(1, diff_nums + 1)
        selected_indices = rng.choice(diff_indices, modified_diff, replace=False)
        new_position = np.copy(position_origin)
        for idx in selected_indices:
            new_position[idx] = 1 - new_position[idx]
        return new_position

    count = 0
    while True:
        position_origin = generate_new_position()

        for i in range(len(position_origin) - 2):
            if position_origin[i] == 1 and position_origin[i + 1] == 1:
                idx_to_flip = rng.choice([i, i + 1])
                position_origin[idx_to_flip] = 0
            elif position_origin[i] == 1 and position_origin[i + 2] == 1:
                idx_to_flip = rng.choice([i, i + 2])
                position_origin[idx_to_flip] = 0

        start_idx, length = find_longest_zero_sequence(position_origin)
        if length >= 5:
            position_origin[int(start_idx + length / 2)] = 1

        if np.sum(position_origin) > min_ones and tuple(position_origin) not in unique_positions:
            unique_positions.add(tuple(position_origin))
            return position_origin
        count += 1
        if count >= 1000:
            print("Warning: Unable to find a suitable position. Exiting function.")
            return None


def save_pareto_to_csv(population, filename='population_results.csv'):
    if not population:
        print("Warning: Population is empty. CSV will only contain headers.")

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    file_path = os.path.join(script_dir, filename)


    fieldnames = ['position', 'dataflow', 'numPEs', 'L1Size', 'L2Size', 'error', 'latency', 'energy']

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
