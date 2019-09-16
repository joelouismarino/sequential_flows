import os, sys

script_path = '/local-scratch/chenleic/Projects/sequential_flows/'

for n_trans in [1, 2]:
    for n_layers in [2]:
        for n_units in [32]:
            for buffer_length in [3]:
                command = 'python main_cedar.py --use_local_path --n_trans {} --n_layers {} --n_units {} --buffer_length {}'.format(
                           n_trans,
                           n_layers,
                           n_units,
                           buffer_length)

                os.system('cd {} && {}'.format(script_path, command))

