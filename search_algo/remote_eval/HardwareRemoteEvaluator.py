import paramiko

class HardwareRemoteEvaluator:
    def __init__(self, remote_host, username, key_file_path):
        self.remote_host = remote_host
        self.username = username
        self.key_file_path = key_file_path

    def generate_hw_file_content(self, num_pes, l1_size, l2_size, noc_bw, noc_num_hops):
        return f"NumPEs: {num_pes}\nL1Size: {l1_size}\nL2Size: {l2_size}\nNoC_BW: {noc_bw}\nNoC_NumHops: {noc_num_hops}\n".replace("'", "'\"'\"'")

    def execute_remote_command(self, hw_params, mapping_file_path):
        with paramiko.SSHClient() as client:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.remote_host, username=self.username, key_filename=self.key_file_path)

            hw_file_content = self.generate_hw_file_content(**hw_params)
            commands = [
                "source /home/anaconda3/etc/profile.d/conda.sh",
                "conda activate maestro-env",
                "cd /home/maestro",
                f"printf '%b' '{hw_file_content}' > /home/maestro/data/hw/temp_hw.m",
                f"./maestro --HW_file='/home/maestro/data/hw/temp_hw.m' --Mapping_file='{mapping_file_path}' --print_res=false --print_res_csv_file=true --print_log_file=false"
            ]
            command = " && ".join(commands)
            docker_command = f"docker exec b80 /bin/bash -c \"{command}\""
            # logging.debug(f"Executed remote command with params: {hw_params}")
            stdin, stdout, stderr = client.exec_command(docker_command)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                print(f"HardwareRemotEvaluator.py -- An error occurred: {error}")
                return False
            return True

    def execute_remote_csv_processing(self, model, dataflow, script_path, exit_layers, exit_counts):
        with paramiko.SSHClient() as client:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.remote_host, username=self.username, key_filename=self.key_file_path)

            exit_layers_str = ','.join(map(str, exit_layers))
            exit_counts_str = ','.join(map(str, exit_counts))
            command = f"docker exec b80 /bin/bash -c 'source /home/anaconda3/etc/profile.d/conda.sh && conda activate maestro-env && python {script_path} --model {model} --dataflow {dataflow} --exit_layers {exit_layers_str} --exit_counts {exit_counts_str}'"

            stdin, stdout, stderr = client.exec_command(command)
            output = stdout.read().decode('utf-8').strip()
            error = stderr.read().decode('utf-8')

            INVALID_SOLUTION_PENALTY = 1e15
            if error and "does not exist" in error:
                print(f"Error: {error.strip()}")
                return INVALID_SOLUTION_PENALTY, INVALID_SOLUTION_PENALTY

            try:
                values = output.split(',')
                return values[0], values[1]
            except ValueError:
                print("Error: Output does not contain enough values.")
