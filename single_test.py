'''
Single run for debugging
'''

from classes.model import Model
from classes.scenario import Scenario

# Scenario overwrites default values
scenario = Scenario({"run_duration": 5})

# Set up model
model = Model(scenario)

# Run model
model.run()