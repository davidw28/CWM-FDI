filename = "ping0001.log"
outputname = "ping0001.txt"

o = open(outputname, "w")

with open(filename,"r") as f:
    for line in f:        
        if line.find("64 bytes from 127.0.0.1") != 0:
            continue
        i = line.find("time=")
        j = line.find(" ms")
        assert i > 0
        assert j > i
        
        o.write(line[i:j].strip().lstrip("time=") + "\n")
        
