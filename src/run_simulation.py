import sys
import numpy as np
import importlib
import importlib.util
import os
import copy
import lxml
import lxml.etree
import cma



sys.path.append(os.path.join(os.path.dirname(__file__), '../rise/build/src/rise'))

# sys.path.append('/home/slk6335/txt2bot/rise/build')
_rs = importlib.import_module("rise")
Rise = _rs.Rise

popsize = 10
configs = []

with open("data/env.rsc", "r") as f:
    env_config = f.read()
with open("data/103.rsc", "r") as f:
    robot = f.read()

for i in range(popsize):
    configs.append([env_config, robot.replace("robot.result", f"r_{i}.result").replace("robot.history", f"r_{i}.history")])

n_rotation_signals = 13

def _read_value(root, p, axis):
    return float(root.xpath(f'//{p}_center_of_mass/{axis}/text()')[0])

def _read_point(root, p):
        return np.array([_read_value(root, p, 'x'),
                         _read_value(root, p, 'y'),
                         _read_value(root, p, 'z')])

def evaluate(summary):
    root = lxml.etree.fromstring(summary.encode())
    start_com = _read_point(root, "start")
    end_com = _read_point(root, "end")
    displacement = end_com - start_com
    return - displacement[0] - 10 * np.min(displacement[2], 0)



def make_actuation(offset):
    def actuation(time, frame, expansion_signals, rotation_signals):
        for i in range(n_rotation_signals):
            rotation_signals[i] = offset[n_rotation_signals + i] * np.sin(time * 8 + offset[i])
    return actuation

def make_actuations(solutions):
    actuations = []
    for i in range(len(solutions)):
        offset = solutions[i]
        actuations.append(make_actuation(offset))
    return actuations

es = cma.CMAEvolutionStrategy(
    [np.pi] * n_rotation_signals + [1.5] * n_rotation_signals,
    1,
    {
        "seed": 626,
        "popsize": popsize,
        "bounds": [
            [0] * n_rotation_signals * 2,
            [2 * np.pi] * n_rotation_signals + [4] * n_rotation_signals,
        ],
        "CMA_stds": [1] * n_rotation_signals + [1] * n_rotation_signals,
    },
)
r = Rise([0], 2048)

offset = np.random.rand(n_rotation_signals * 2)
best_offset = copy.deepcopy(offset)
best = float('inf')
gen = 0
while not es.stop():
    print(gen)
    solutions = es.ask()
    actuations = make_actuations(solutions)
    validity = r.run_sims(configs, actuations, record_buffer_size=300, policy="batched")
    fitness = np.zeros(popsize)
    for i in range(popsize):
        if validity[i]:
            with open(f"r_{i}.result", "r") as f:
                summary = f.read()
            fitness[i] = evaluate(summary)
    print(fitness)
    if np.min(fitness) < best:
        best_index = np.argmin(fitness)
        best = fitness[best_index]
        best_offset = copy.deepcopy(solutions[best_index])
        print(best_offset)
        os.rename(f"r_{best_index}.history", f"result/{gen}_{best}.history")
    es.tell(solutions, fitness)
    es.logger.add()
    es.disp()
    gen += 1

offset = best_offset