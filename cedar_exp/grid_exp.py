import os, sys

job_script = '''#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=def-mori
#SBATCH --output=/home/arzela/jobs/logs/{}

source ~/py3/bin/activate
cd /home/arzela/scratch/sequential_flows'''

job_root = '/home/arzela/scratch/jobs/sequential_flow_grid'

for n_trans in [2, 3]:
    for n_layers in [2, 4]:
        for n_units in [32, 64]:
            for buffer_length in [1,3,5]:
                # if n_layers == 2 and n_units == 32 and buffer_length == 3:
                #     continue

                command = 'python main_cedar.py --n_trans {} --n_layers {} --n_units {} --buffer_length {}'.format(
                           n_trans,
                           n_layers,
                           n_units,
                           buffer_length)

                log_name = 'n_trans_{}_n_layers_{}_n_units_{}_buffer_length_{}.log'.format(
                            n_trans,
                            n_layers,
                            n_units,
                            buffer_length)

                job_script = job_script.format(log_name) + '\n' + command
                # print("""asfsafasdfsdfdf""")
                # input(job_script)

                file_name = log_name[:-4] + '.sh'
                file_name = os.path.join(job_root, file_name)
                with open(file_name, 'w') as f:
                    f.write(job_script)

                sbatch_command = 'cd {} && sbatch {}'.format(job_root, file_name)
                os.system(sbatch_command)

                # sys.exit()