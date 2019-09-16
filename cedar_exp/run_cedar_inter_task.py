import sys, os
import itertools

task_id = int(sys.argv[1])

trans_list = [3]
buffer_list = [1,3,5]

if task_id == 0:
    unit_list = [32]
    layer_list = [2]
elif task_id == 1:
    unit_list = [32]
    layer_list = [4]
elif task_id == 2:
    unit_list = [64]
    layer_list = [2]
elif task_id == 3:
    unit_list = [64]
    layer_list = [4]
else:
    print('invalid task id')
    sys.exit()

script_root = '/home/arzela/scratch/sequential_flows/'

for n_trans, n_layers, n_units, buffer_length in itertools.product(trans_list, layer_list, unit_list, buffer_list):

    command = 'python main_cedar.py --n_trans {} --n_layers {} --n_units {} --buffer_length {}'.format(
               n_trans,
               n_layers,
               n_units,
               buffer_length)

    srun_command = 'cd {} && {}'.format(script_root, command)
    os.system(srun_command)

    # sys.exit()