Quickstart
==========

Working on iPython
##################

Log manipulation
****************

Aggregating logs from different experiments
-------------------------------------------

Suppose you have the logs of two experiments 1 and 2 on the default log directory::

    log/
    ├─ experiment_1/
    │  ├─ reactor_1.csv
    │  ├─ reactor_2.csv
    │  ├─ notes.txt
    ├─ experiment_2/
    │  ├─ reactor_1.csv
    │  ├─ reactor_2.csv
    │  ├─ misc.html

To simply join the contents of the csv's for each reactor in a folder `experiment_aggregated`, do the following in a iPython console:

:: python

    from pyduino.log import LogAggregator
    lagg = LogAggregator(["log/experiment_1/*.csv","log/experiment_2/*.csv"])
    lagg.agg("experiment_aggregated/")

Extracting particular columns from a log folder
-----------------------------------------------

The method `pyduino.log.log.transpose` can be used to extract specific columns from all reactors in
a given experiment and save them into a separate folder having the csv's of each column in a different
file. Using the same example as above, if we wish to extract `column_1` and `column_2` from all reactors
in `experiment_1` into a new folder `experiment_transpose`, we do as follows:

:: python

    from pyduino.log import log
    log1 = log("*.csv",name="experiment_1")
    log1.transpose(["column_1","column_2"],"experiment_transpose")

In both of the aforementioned options, you can pass an integer to the parameter `skip` to jump a certain number
of lines instead of reading the entire csv. Conceptually, this is the same as slicing a list with `::skip`.
So, if you pass zero, no lines would be read.