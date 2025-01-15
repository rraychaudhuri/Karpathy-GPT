import os
import logging
import configparser



def readConfig(projectRoot=None):
    """
    Read the config file and return a config object
    """
    if projectRoot is None:
        projectRoot = os.getcwd()

    config = configparser.ConfigParser()
    configfile = os.path.join(os.path.join(projectRoot, "config"), "main.cfg")
    if not os.path.exists(configfile):
        raise FileNotFoundError(f"No config file found at : {configfile}")

    config.read(configfile)
    return config


def loadData(projectRoot=None):
    """
    Create a config opbject and read the datafile
    """
    config = readConfig(projectRoot)
    datafile = os.path.join(config["InputData"]["datadir"], config["InputData"]["fname"])
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"No file found at : {datafile}")

    all_data = ""    
    with open(datafile, "r") as fp:
        all_data = "".join(fp.readlines())

    return all_data


def foo():
    print("Hello World !!")


if __name__ == '__main__':
    all_data = loadData()
    print(all_data[:100])
