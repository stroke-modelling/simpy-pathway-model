import os
os.chdir('../')

from classes.replication import Replicator
from classes.parameters import Scenario

scenarios = {}

scenarios['default'] = Scenario(
    {"run_duration": 5})

scenarios['fast_ambo'] = Scenario({
    "run_duration": 5,
    "process_time_ambulance_response": (0, 5),
    "process_time_call_ambulance": (0, 5)})

# Set up and call replicator
replications = Replicator(scenarios=scenarios, replications=5)
replications.run_scenarios()
