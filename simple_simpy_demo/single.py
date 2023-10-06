from sim_classes.model import Model
from sim_classes.parameters import Scenario

scenario = Scenario()

model = Model(scenario)
model.run()

# Print results
print ('Bed and queue numbers\n')
print (model.results_audit)
print ('\nQueue times\n')
print (model.results_qtime)

