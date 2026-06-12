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
    time_py = None
    rmse = None
    pearsonr = None

    for line in lines:
        if "Time=" in line:
            time_py = float(line.lstrip("Time=").rstrip("s"))
        if "RMSE=" in line:
            rmse = float(line.lstrip("RMSE="))
        elif "PearsonR=" in line:
            pearsonr = float(line.lstrip("PearsonR="))

    return time, energy, time_py, rmse, pearsonr

def measure_model(train_path, test_path, model_name):
    model_path = f"models/{model_name}.pkl"
    results = {}
    
    print(f"===TRAINING {model_name.upper()}===")
    result = call_python(f"train_model.py {train_path} {model_path}", turbostat = True)
    time, energy, time_py, _, _ = extract_result(result)
    results["Training"] = (energy, time_py)
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s")
    print()
    
    print(f"===CALLING {model_name.upper()} (TRAIN) ===")
    result = call_python(f"call_model_batch.py {train_path} {model_path}", turbostat = True)
    time, energy, time_py, rmse, pearsonr = extract_result(result)
    results["Calling (train)"] = (energy, time_py, rmse, pearsonr)
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s RMSE={rmse} PearsonR={pearsonr}")
    print()

    print(f"===CALLING {model_name.upper()} (TEST) ===")
    result = call_python(f"call_model_batch.py {test_path} {model_path}", turbostat = True)
    time, energy, time_py, rmse, pearsonr = extract_result(result)
    results["Calling (test)"] = (energy, time_py, rmse, pearsonr)
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s RMSE={rmse} PearsonR={pearsonr}")
    print()

    return results

def write_results(results, out_path):
    with open(out_path, "w") as f:
        print(results, file = f)

OUT_DIR = "data/"

TRAIN_PATH = "data/wine_train.csv"
TEST_PATH = "data/wine_test.csv"
_ = call_python(f"split_wine_data.py {TRAIN_PATH} {TEST_PATH}")

model_names = (
    "model01",
    "model02",
)

for model_name in model_names:
    results = measure_model(TRAIN_PATH, TEST_PATH, model_name)
    write_results(results, "results/wine_{model_name}.txt")