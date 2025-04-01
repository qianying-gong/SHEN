import sys
import aux_funcs as af
import model_funcs as mf
import network_architectures as arcs
from profiler import profile_sdn, profile

def early_exit_experiments(dataset, model, models_path, position, device='cpu'):
    sdn_training_type = 'sdn_training'

    task = dataset

    sdn_name = model + '_sdn'
    sdn_name = task + '_' + sdn_name + '_' + sdn_training_type

    print(sdn_name)


    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)

    # to test early-exits with the SDN
    one_batch_dataset = af.get_dataset(sdn_params['task'], 1)
    total_ops, total_params = profile_sdn(sdn_model, sdn_model.input_size, device)
    print("#Ops (G): {}".format(total_ops))

    # search for the confidence threshold for early exits
    confidence_thresholds = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.95, 0.99, 0.999, 0.9]
    sdn_model.forward = sdn_model.early_exit

    for threshold in confidence_thresholds:
        print(threshold)
        sdn_model.confidence_threshold = threshold

        # change the forward func for sdn to forward with cascade
        top1_test, early_exit_counts, non_conf_output_counts = mf.sdn_test_early_exits(sdn_model, one_batch_dataset.test_loader, device)
        average_mult_ops = 0
        total_num_instances = 0

        # 早期退出计数
        for output_id, output_count in enumerate(early_exit_counts):
            average_mult_ops += output_count * total_ops[output_id]
            total_num_instances += output_count

        # 非置信退出计数
        for output_count in non_conf_output_counts:
            total_num_instances += output_count
            average_mult_ops += output_count * total_ops[output_id]

        average_mult_ops /= total_num_instances

        early_exit_counts[-1] = 10000 - sum(early_exit_counts[:-1])

        af.save_position_error_to_csv(position, threshold, average_mult_ops, 100.0 - top1_test, early_exit_counts, non_conf_output_counts)

        if threshold == 0.9:
            return early_exit_counts, 100.0 - top1_test


class DualLogger(object):
    def __init__(self, filepath, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()