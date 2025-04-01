import argparse
import math
import time
import config
import EENNO.EarlyExitNetwork.network_architectures as arcs
from search_space.eex_hw_search_space import EExHardwareSearchSpace
from w_maestro.frameworks_to_modelfile_maestro import convert_to_model
from w_maestro.modelfile_to_mapping import model_to_mapping
from utils_opt import RankAndCrowdingSurvivalInner, filter_population, RankAndCrowdingSurvivalRank0Only
from utils_eval import remote_eval_hardware
from utils_pso import *

parser = argparse.ArgumentParser(description='Run the optimization framework of EENNO for optimal DyNNs')
parser.add_argument('--config-file', default='./config_evo.yml')
run_args = parser.parse_args()

def nsga_optimization(pareto_overall, position, exit_counts, error, args):
    """

    :param position: position with IC search space, a list with 1*n_dimensions
    :param exit_counts: exit samples in each IC
    :param error: the overall error rate of early exit network
    """
    rng = np.random.default_rng()

    # construct model
    sdn_name = str(args.dataset) + '_' + str(args.model) + '_sdn_sdn_training'
    print(sdn_name)
    models_path = ''.format(af.get_random_seed())
    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    sdn_model.to(device)
    exit_layers = convert_to_model(sdn_params, sdn_model, outfile=str(args.model)+'ee.m')
    model_to_mapping(model=args.model)

    # DNN Accelerator hardware search space
    ss_eex_hardware = EExHardwareSearchSpace()
    parent_popu_nsga = ss_eex_hardware.initialize_all(args.nsga.parent_popu_size)

    pareto_inner = {}

    for evo in range(args.nsga.evo_iter):

        remote_eval_hardware(args.model, parent_popu_nsga, position, exit_layers, exit_counts, error)

        # Survival selection for optimization based on eex_hardware performances:
        nb_survival = math.ceil(len(parent_popu_nsga) * args.nsga.survival_ratio)
        parent_popu_nsga = filter_population(parent_popu_nsga)
        survivals = RankAndCrowdingSurvivalInner(pop=parent_popu_nsga, n_survive=nb_survival)

        # Update the Pareto frontier
        for i, cfg in enumerate(survivals, start=0):
            pareto_inner[i] = cfg
            pareto_overall.append(cfg)

        # Create the next batch of eex_dvfs to evaluate
        parent_popu_nsga = []

        # Crossover
        for _ in range(args.nsga.crossover_size):
            cfg1 = rng.choice(list(pareto_inner.values()))
            cfg2 = rng.choice(list(pareto_inner.values()))
            cfg = ss_eex_hardware.crossover_and_reset(cfg1, cfg2, args.nsga.crossover_prob)
            parent_popu_nsga.append(cfg)

        # Mutation
        for _ in range(args.nsga.mutate_size):
            old_cfg = rng.choice(list(pareto_inner.values()))
            cfg = ss_eex_hardware.mutate_and_reset(old_cfg, prob=args.nsga.mutate_prob)
            parent_popu_nsga.append(cfg)


def pso_optimization(args):

    positions = np.random.randint(2, size=(args.pso.n_particles-1, args.pso.n_dimensions))
    positions = adjust_initial_positions(positions, args.pso.min_ones, args.pso.max_ones)

    # insert initial paritcles' positions into set for filter same position
    unique_positions = set()
    for pos in positions:
        position_tuple = tuple(pos)
        unique_positions.add(position_tuple)

    pso_set = [{'position': pos} for pos in positions]

    pareto_overall = []
    for particle in pso_set:
        position = particle.get('position')

        particle['personal_best_position'] = position
        exit_counts, error = fit_error(args.dataset, args.model, position)
        particle['exit_counts'] = exit_counts
        particle['error'] = error

        particle['personal_best_error'] = error
        nsga_optimization(pareto_overall, position, exit_counts, error, args)

    global_best_particle = min(pso_set, key=lambda x: x['error'])
    global_best_position = global_best_particle['position']
    global_best_error = global_best_particle['error']


    for iteration in range(args.pso.n_iterations):
        for individual in pso_set:

            position_origin = individual.get('position')
            personal_best_position = individual.get('personal_best_position')
            personal_best_error = individual.get('personal_best_error')
            position = adjust_single_position(unique_positions, personal_best_position, position_origin, args.pso.min_ones)
            if position is None:
                continue
            position = adjust_single_position(unique_positions, global_best_position, position, args.pso.min_ones)
            if position is None:
                continue
            exit_counts, error = fit_error(args.dataset, args.model, position)

            if error < personal_best_error:
                individual['personal_best_position'] = position.copy()
                individual['personal_best_error'] = error
                nsga_optimization(pareto_overall, position, exit_counts, error, args)

            if error < global_best_error:
                global_best_position = position
                global_best_error = error

    pareto_overall = filter_population(pareto_overall)

    pareto_particles = RankAndCrowdingSurvivalRank0Only(pop=pareto_overall)
    print('pareto_particles')
    print(pareto_particles)
    save_pareto_to_csv(pareto_particles)



if __name__ == '__main__':
    start_time = time.time()
    af.set_random_seeds()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = config.setup(run_args.config_file)
    pso_optimization(args=args)
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"Running took {hours} hours, {minutes} minutes.")