# Getting started

API documentation: [pyduino-parallel](https://fotobiolab-unb.github.io/pyduino-parallel/pyduino.html)

## Reactor Manager
### Basic usage

Run:
- `ipython3`
- `run pyduino2`

Check reactors:
- `r.reactors`

Manual connect to all reactors (unnecessary):
- `r.send("manual_connect")`

Manual connect to reactor number X:
- `r.reactors[X].connect()`
- `r.reactors[X].send("manual_connect")`

Set parameters to all reactors:
- `r.send("set(branco,100,brilho,100)",False)`

Set parameters to reactor X:
- `r.reactors[X].set({"branco":100,"brilho":100})`
- `r.reactors[X]._send("set(branco,100,brilho,100)")`

Set parameters to all reactors:
- `r.send("set(branco,100,full,0)",False)`

Get data from all reactors:
- `r.send("dados")`

Get data from reactor X:
- `r.reactors[X].send("dados")`
- `r.reactors[X].get()`

Log all data to disk:
- `r.log_dados()`

Create a new log folder:
- `r.log_init()`

### Run calibration

- `r.calibrate()`

## Genetic Algorithm

# Basic Usage

Run loop without genetic algorithm for 5 seconds:
- `g.run(5,run_ga=False)`

Run a command repeatedly:
- `for i in range(4): g.send("drenar",False)`