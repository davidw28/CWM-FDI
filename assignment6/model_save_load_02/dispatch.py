import subprocess

PYTHON3 = "/home/ubuntu/CWM-FDI/assignment6/.venv/bin/python3"

def call_python(py_path, turbostat = False, stdout = False, stderr = False):
    cmd = f"{PYTHON3} {py_path}"
    if turbostat:
        cmd = "sudo turbostat -q -S --Joules --show Pkg_J " + cmd
        
    result = subprocess.run(cmd.split(" "), capture_output = True, text = True)
    
    if stdout:
        print(result.stdout.strip())
        
    if stderr:
        print(result.stderr.strip())
        
    return result

def extract_result(result):
    # stdout contains information printed by run_model.py
    # stderr contains turbostat output
    
    ## result.stderr is in the form '1.331949 sec\nPkg_J\n30.90\n'
    lines = result.stderr.strip().split() # form ['1.203377', 'sec', 'Pkg_J', '17.68']

    assert len(lines) == 4
    assert lines[1] == 'sec'
    assert lines[2] == 'Pkg_J'
    
    time = float(lines[0])
    energy = float(lines[3])

    ## result.stdout is in the form 'RMSE=0.1027\nPearsonR=0.737\n'
    lines = result.stdout.strip().split() # form ['RMSE=0.1027', 'PearsonR=0.737']
    rmse = None
    pearsonr = None

    for line in lines:
        if "RMSE=" in line:
            rmse = float(line.lstrip("RMSE="))
        elif "PearsonR=" in line:
            pearsonr = float(line.lstrip("PearsonR="))

    return time, energy, rmse, pearsonr
    
OUT_DIR = "data/"

_ = call_python("split_data.py")

print("===TRAINING MODEL===")
result = call_python("train_model.py data/wine_train.csv models/model_1.pkl", turbostat = True)
time, energy, _, _ = extract_result(result)
print(f"Time={time} s Energy={energy} J")
print()

print("===CALLING MODEL (TRAIN) ===")
result = call_python("call_model_batch.py data/wine_train.csv models/model_1.pkl", turbostat = True)
time, energy, rmse, pearsonr = extract_result(result)
print(f"Time={time} s Energy={energy} J RMSE = {rmse} PearsonR={pearsonr}")
print()

print("===CALLING MODEL (TEST)===")
result = call_python("call_model_batch.py data/wine_test.csv models/model_1.pkl", turbostat = True)
time, energy, rmse, pearsonr = extract_result(result)
print(f"Time={time} s Energy={energy} J RMSE = {rmse} PearsonR={pearsonr}")
print()
