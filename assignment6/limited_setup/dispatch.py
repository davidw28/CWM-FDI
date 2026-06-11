import subprocess

PY_PATH = "../../assignment1/matmul_fast.py"
VENV_PATH = "/home/ubuntu/CWM-FDI/assignment6/.venv"
# OUT_PATH = "./data/myturbostat.log"

cmd = f"{VENV_PATH}/bin/python3 run_model.py"

result = subprocess.run(cmd.split(" "), capture_output = True, text = True)
print("Complete")
print(result.stderr)