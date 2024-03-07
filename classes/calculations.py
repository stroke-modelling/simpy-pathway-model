"""
Units class for defining choice of stroke unit through the pathway.

TO DO - should this be a functions file?
don't want a big run() function here - have clear names for what it actually does.
Yes - the only self. remaining are Setup, can replace with direct paths/filenames as kwargs.
Plus read in files using import_relative or whatever, 
do it in the Scenario class, pass the df to here as arg.

"""
import pandas as pd
import os  # For checking directory existence

from classes.setup import Setup


class Calculations(object):
    """
    Links between stroke units.
    """

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # If no setup was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()

    # ################
    # ##### LSOA #####
    # ################



    # #################
    # ##### UNITS #####
    # #################