from sim_classes.replication import Replicator
from sim_classes.parameters import Scenario

scenarios = {}

scenarios['default'] = Scenario()
scenarios['low_demand'] = Scenario(interarrival_time=0.07)

# Set up and call replicator
replications = Replicator(scenarios=scenarios, replications=8)
replications.run_scenarios()