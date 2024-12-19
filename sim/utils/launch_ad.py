import subprocess
import time
import os


def launch(shell_path, cuda_id, output):
    os.makedirs(output, exist_ok=True)
    print(os.path.join(output, 'output.txt'))
    print(shell_path, cuda_id, output)
    with open(os.path.join(output, 'output.txt'), 'w') as f:
        process = subprocess.Popen(
            ["zsh", shell_path, cuda_id, output], stdout=f, stderr=f
        )
    return process


def check_alive(process, tolerant=100):
    i = 0
    while i < tolerant:
        return_code = process.poll()
        if return_code is not None:
            print(f"The AD algorithm completed with return code {return_code}.")
            process.kill()
            return
        elif i % 5 == 0:
            print(f"The AD algorithm is still running, remaining tolerant {tolerant - i}.")
        time.sleep(1)
        i += 1
    process.kill()
    print("The AD algorithm process is killed.")