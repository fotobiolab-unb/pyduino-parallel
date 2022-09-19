Configuration Files
===================

The configuration files are in YAML format and are divided in four sections: hyperparameters, slave, system and dash. By default, `init_spectra.py` will look for the default confguration file `config.yaml` located inside the `pyduino` folder. In order to specify a custom file, pass the flag `--config` along with the path. So, for a file named `custom_config_file.yaml`, the syntax will be

>>> run init_spectra.py --config custom_config_file.yaml


Slave Parameters
----------------

The slave section gets parameters which are responsible to deal with the connection with the reactors. As the newtork search is done with Nmap, you can use the same conventions in the inputs.

- `ports`: The port in which the slave service is running in the reactor. It's 5000 by default.
- `network`: Network to scan for reactors. As this will be passed to Nmap, you can specify a single IP, CIDR notation (192.168.1.0/24), or a range like "192.168.1.0-255".
- `exclude`: IPs to ignore in the search; these won't be used by the program. For multiple IPs, use a comma to separate them: `'x.x.x.x,y.y.y.y'`. If there are no IPs to exclude, just pass `false` as a value.
