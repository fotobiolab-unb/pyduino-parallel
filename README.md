# Getting started
## Basic usage

Run:
- `ipython3`
- `run pyduino2`

Check reactors:
- `r._id`

Manual connect to all reactors (unnecessary):
- `r.send("manual_connect")`

Manual connect to reactor number X:
- `r.reactors[r._id[X]].connect()`
- `r.reactors[r._id[X]].send("manual_connect")`

Set parameters to all reactors:
- `r.send("set(branco,100,brilho,100)")`

Set parameters to reactor X:
- `r.reactors[r._id[X]].set({"branco":100,"brilho":100})`
- `r.reactors[r._id[X]].send("set(branco,100,brilho,100)")`

Set parameters to all reactors:
- `r.send("set(branco,100,full,0)",False)`

Get data from all reactors:
- `r.send("dados")`

Get data from reactor X:
- `r.reactors[r._id[X]].send("dados")`
- `r.reactors[r._id[X]].get()`

Log all data to disk:
- `r.log_dados()`

Create a new log folder:
- `r.log_init()`

## Run calibration

- `r.calibrate()`