Logging
=======

The submodule `log` is responsible to backup row-formatted data into the disk. By default, it is implemented as a subclass of `ReactorManager` and can be accessed as an attribute like in the following example:

.. code-block:: python

   r = ReactorManager()
   r.log

As a standalone class, `log` takes three parameters: `subdir`, `path`, and `name`. Only the parameter `subdir` must be specified with a list of strings, whose names are the file names for the CSV's where the row data will be saved. Even though the parameter `name` expects a string to use as folder name, passing `None` to it will set the current timestamp instead.

When instantiated from within the `Spectra` class, `log` automatically uses a list of the reactors' ids as input for the `subdir` parameter.

If a CSV file already exists, `log` will append the new data instead of overwriting the file.

Caching
-------

To quickly retrieve the last parameter values sent to the reactors in the past and also account for possible unforeseen interruptions, the `log` module possesses a caching mechanism implemented in the `cache_data` method. This method is triggered at every call of `ReactorManager.dados` by default and will store the output of all columns from all reactors as a single file.

Regarding `log` as a subclass of `ReactorManager`, values from the cached file (or any other initialization file with the same schema) can be reapplied into the reactors through the `set_preset_state` method by setting the parameter `path` as the path to the cached file as follows:

.. code-block:: python

   #Restoring a previous state from a cache file
   r.set_preset_state(path="path_to_cached_file.csv")