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

def extract_time_energy_RMSE(result):
    # stdout contains information printed by run_model.py
    # stderr contains turbostat output
    # For now, RMSE = None
    
    """result.stderr is in the form '1.331949 sec\nPkg_J\n30.90\n'"""
    
    lines = result.stderr.strip().split()
    ## This should be in the form ['1.203377', 'sec', 'Pkg_J', '17.68']
    
    assert len(lines) == 4
    assert lines[1] == 'sec'
    assert lines[2] == 'Pkg_J'
    
    time = float(lines[0])
    energy = float(lines[3])
    
    return time, energy, None
    
_ = call_python("split_data.py")
result = call_python("run_model.py", turbostat = True)
time, energy, RMSE = extract_time_energy_RMSE(result)
print(f"Time={time} s Energy={energy} J RMSE = {RMSE}")
