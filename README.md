# Getting started
## Basic usage

Run:
- `ìpython3`
- `run pyduino2`

Check reactors:
- `reactors`

Manual connect to all reactors:
- `func("manual_connect")`

Manual connect to reactor number X:
- `reactors[df[X]].connect()`
- `reactors[df[X]].send("manual_connect")`

Set parameters to all reactors:
- `func("set(branco,100,brilho,100)")`

Set parameters to reactor X:
- `reactors[df[X]].set({"branco",100,"brilho",100})`

Get data from all reactors:
- `func("dados")`

Get data from reactor X:
- `reactors[df[X]].send("dados")`
- `reactors[df[X]].get()`

