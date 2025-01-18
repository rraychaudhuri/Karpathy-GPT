import os
import configparser


class DataUtils:
    def __init__(self, projectRoot=None):
        self.projectRoot = projectRoot if projectRoot is not None else os.getcwd()

        self.all_data = None
        self.encode = None
        self.decode = None
        self.vocab = None
    
    def initialize(self):
        """
        Create a config opbject and read the datafile
        """
        config = DataUtils.readConfig(self.projectRoot)
        datafile = os.path.join(config["InputData"]["datadir"], config["InputData"]["fname"])
        if not os.path.exists(datafile):
            raise FileNotFoundError(f"No file found at : {datafile}")

        self.all_data = ""    
        with open(datafile, "r") as fp:
            self.all_data = "".join(fp.readlines())
        
        self.vocab = set("".join(self.all_data))
        
        stoi = {ch:i for i, ch in enumerate(sorted(self.vocab))}
        itos = {v:k for k,v in stoi.items()}
        self.encode = lambda s: stoi[s] if len(s) < 2 else [stoi[ch] for ch in s]
        self.decode = lambda item: itos[item] if isinstance(item, list) is False else "".join([itos[i] for i in item])

    @classmethod
    def readConfig(cls, projectRoot=None):
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




