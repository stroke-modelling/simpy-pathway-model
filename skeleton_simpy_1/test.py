from classes.model import Model


# Scenario overwrites default values
scenario = {"run_duration": 5 * 1440}

model = Model(scenario)

model.run()
