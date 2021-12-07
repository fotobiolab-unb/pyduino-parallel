# Getting started
## Basic usage

Run:
- `ìpython3`
- `run pyduino2`

Check reactors:
- `r._id`

Manual connect to all reactors (unnecessary):
- `r.send("manual_connect")`

Manual connect to reactor number X:
- `r.reactors[r._id_reverse[X]].connect()`
- `r.reactors[r._id_reverse[X]].send("manual_connect")`

Set parameters to all reactors:
- `r.send("set(branco,100,brilho,100)")`

Set parameters to reactor X:
- `r.reactors[r._id_reverse[X]].set({"branco":100,"brilho":100})`
- `r.reactors[r._id_reverse[X]].send("set(branco,100,brilho,100)")`

Get data from all reactors:
- `r.send("dados")`

Get data from reactor X:
- `r.reactors[r._id_reverse[X]].send("dados")`
- `r.reactors[r._id_reverse[X]].get()`

Log all data to disk:
- `r.log_dados()`

