import numpy as np
import copy


class EExHardwareSearchSpace:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.dataflow = ['os', 'rs', 'ws']
        self.numPEs = [32, 64, 128, 168, 256, 512, 1024]
        self.L1Size = [512, 1024, 2048, 3072]
        self.L2Size = [55296, 65536, 110592, 159744, 1048576, 2097152, 4194304]

        self.cfg_candidates = {
            'dataflow': self.dataflow,
            'numPEs': self.numPEs,
            'L1Size': self.L1Size,
            'L2Size': self.L2Size
        }

    def sample_all(self, n_samples=1):

        data = []
        for _ in range(n_samples):

            dataflow_ = self.rng.choice(self.cfg_candidates['dataflow'])
            numPEs_ = self.rng.choice(self.cfg_candidates['numPEs'])
            L1Size_ = self.rng.choice(self.cfg_candidates['L1Size'])
            L2Size_ = self.rng.choice(self.cfg_candidates['L2Size'])

            data.append({'dataflow': dataflow_, 'numPEs': numPEs_, 'L1Size': L1Size_, 'L2Size': L2Size_})

        return data


    def initialize_all(self, n_doe):
        return self.sample_all(n_samples=n_doe)

    # *************************************************** Mutation and Crossover ******************************************************#
    def mutate_and_reset(self, cfg, prob=0.1):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: x if len(candidates) == 1 else self.rng.choice(
            [v for v in candidates if v != x])

        r = self.rng.random()
        if r < prob:
            cfg['dataflow'] = pick_another(cfg['dataflow'], self.cfg_candidates['dataflow'])

        if r < prob:
            cfg['numPEs'] = pick_another(cfg['numPEs'], self.cfg_candidates['numPEs'])

        if r < prob:
            cfg['L1Size'] = pick_another(cfg['L1Size'], self.cfg_candidates['L1Size'])

        if r < prob:
            cfg['L2Size'] = pick_another(cfg['L2Size'], self.cfg_candidates['L2Size'])

        return cfg

    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if self.rng.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if self.rng.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}

        # sample a Hardware configuration
        cfg['dataflow'] = cfg1['dataflow'] if self.rng.random() < p else cfg2['dataflow']
        cfg['numPEs'] = cfg1['numPEs'] if self.rng.random() < p else cfg2['numPEs']
        cfg['L1Size'] = cfg1['L1Size'] if self.rng.random() < p else cfg2['L1Size']
        cfg['L2Size'] = cfg1['L2Size'] if self.rng.random() < p else cfg2['L2Size']

        return cfg