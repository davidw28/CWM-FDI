import subprocess
import os
import sys

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

    assert len(lines) == 4, result.stderr
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

def measure_model(train_path, test_path, sample_path, dataset, feature_path, model_name):
    model_path = f"models/{dataset}_{model_name}.pkl"
    results = {}
    
    print(f"=TRAINING=")
    result = call_python(f"train_{model_name}.py {train_path} {model_path} {dataset} {feature_path}", turbostat = True)
    time, energy, time_py, _, _ = extract_result(result)
    results["Training"] = {
        "Energy (J)": energy,
        "Time (s)": time_py,
    }
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s")
    print()
    
    print(f"=CALLING (TRAIN DATA)=")
    result = call_python(f"call_model_batch.py {train_path} {model_path} {dataset} {feature_path}", turbostat = True)
    time, energy, time_py, rmse, pearsonr = extract_result(result)
    results["Calling (train)"] = {
        "Energy (J)": energy,
        "Time (s)": time_py,
        "RMSE": rmse,
        "PearsonR": pearsonr
    }
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s RMSE={rmse} PearsonR={pearsonr}")
    print()

    print(f"=CALLING (TEST DATA)=")
    result = call_python(f"call_model_batch.py {test_path} {model_path} {dataset} {feature_path}", turbostat = True)
    time, energy, time_py, rmse, pearsonr = extract_result(result)
    results["Calling (test)"] = {
        "Energy (J)": energy,
        "Time (s)": time_py,
        "RMSE": rmse,
        "PearsonR": pearsonr
    }
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s RMSE={rmse} PearsonR={pearsonr}")
    print()

    print(f"=CALLING (SAMPLE DATA)=")
    result = call_python(f"call_model_batch.py {sample_path} {model_path} {dataset} {feature_path}", turbostat = True)
    time, energy, time_py, rmse, pearsonr = extract_result(result)
    results["Calling (sample)"] = {
        "Energy (J)": energy,
        "Time (s)": time_py,
        "RMSE": rmse,
        "PearsonR": pearsonr
    }
    print(f"Energy={energy} J Time={time} s Time_py={time_py} s RMSE={rmse} PearsonR={pearsonr}")
    print()

    return results

def write_results(results, out_path):
    with open(out_path, "w") as f:
        print(results, file = f)

# Hardcoded
PYTHON3 = "/home/ubuntu/CWM-FDI/assignment6/.venv/bin/python3"
SKIP = True # Skip if file already exists

# Read command line arguments
assert len(sys.argv) == 2
DATASET = sys.argv[1]

# Prepare data
TRAIN_PATH = f"data/{DATASET}_train.csv"
TEST_PATH = f"data/{DATASET}_test.csv"
SAMPLE_PATH = f"data/{DATASET}_sample.csv"
FEATURE_PATH = f"data/{DATASET}_features.txt"
PYTHON_SPLIT_DATA_PATH = f"split_{DATASET}_data.py"

if SKIP and os.path.isfile(TRAIN_PATH) and os.path.isfile(TEST_PATH) and os.path.isfile(SAMPLE_PATH) and os.path.isfile(FEATURE_PATH):
    print(f"===DATA ALREADY PROCESSED===")
else:
    print(f"===PROCESSING DATA===")
    call_python(f"{PYTHON_SPLIT_DATA_PATH} {TRAIN_PATH} {TEST_PATH} {SAMPLE_PATH} {FEATURE_PATH}")

# Run models
model_names = sorted(x[6:-3] for x in os.listdir() if x[:6] == "train_" and x[-3:] == ".py")

for model_name in model_names:
    try:
        if SKIP and os.path.isfile(f"results/{DATASET}_{model_name}.txt"):
            print(f"===SKIPPING {model_name.upper()}===")
        else:
            print(f"===PROCESSING {model_name.upper()}===")
            results = measure_model(TRAIN_PATH, TEST_PATH, SAMPLE_PATH, DATASET, FEATURE_PATH, model_name)
            write_results(results, f"results/{DATASET}_{model_name}.txt")
    except Exception as e:
        print(f"===FAILED {model_name.upper()}===")
        print(e)
