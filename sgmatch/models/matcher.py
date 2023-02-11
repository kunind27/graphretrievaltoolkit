from typing import Type
from sgmatch.utils.utility import Namespace

# Importing Graph Similarity Models
from NeuroMatch import SkipLastGNN
from SimGNN import SimGNN
from GMN import GMNEmbed, GMNMatch
from GraphSim import GraphSim
from ISONET import ISONET

class graphMatcher():
    """
    A Wrapper Class for all the Graph Similarity / Matching models implemented in the library
    
    Args:
        args (object): args is an object of class 'Namespace' containing arguments to be passed to models
    
    Returns:
        self.model: The initialized model selected by the user through the 'model_name' key in dict 'args'
    """
    def __init__(self, av: Type[Namespace]):
        self.av = av
        return self.graph_match_model(self.av)

    def graph_match_model(self, av: Type[Namespace]):
        self.model = None
        if av.model_name == 'NeuroMatch':
            model = SkipLastGNN(av)
        elif av.model_name == 'SimGNN':
            model = SimGNN(av)
        elif av.model_name == 'GMNEmbed':
            model = GMNEmbed(av)
        elif av.model_name == 'GMNMatch':
            model = GMNMatch(av)
        elif av.model_name == 'GraphSim':
            model = GraphSim(av)
        elif av.model_name == 'ISONET':
            model = ISONET(av)
        else:
            raise NotImplementedError("The model name is incorrect, please use the correct model name")

        return self.model