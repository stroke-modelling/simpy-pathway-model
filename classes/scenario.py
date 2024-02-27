"""
Scenario class with global parameters for the pathway model.

TO DO -----------------------------------------------------------------------------
- write this docstring
- only save inputs to Scenario() for init?

Usually the scenario.yml file should contain:

'name'  # this scenario
# Geography setup: for LSOA, region, units...
'limit_to_england',  # bool
'limit_to_wales',  # bool
'lsoa_catchment_type',  # 'island' or 'nearest'

"""
import numpy as np
import pandas as pd
import os
import yaml

from classes.calculations import Calculations
from classes.setup import Setup


class Scenario(object):
    """
    Global variables for model.

    class Scenario():

    Attributes
    ----------

    hospitals:
        Info on stroke hospitals. Pandas DataFrame.

    inter_arrival_time:
        Time (minutes) between arrivals. Decimal.

    limit_to_england:
        Limit model to only England admissions. Boolean

    lsoa_names:
        List of LSOA names. List.

    lsoa_relative_frequency:
        Relative frequency of admissions to each LSOA (sums to 1). NumPy array.

    lsoa_ivt_travel_time:
        Travel time (minutes) from LSOA to closest IVT unit. Dictionary.

    lsoa_ivt_unit:
        Closest IVT unit postcode for each LSOA. Dictionary.

    lsoa_mt_travel_time:
        Travel time (minutes) from LSOA to closest MT unit. Dictionary.

    lsoa_mt_unit:
        Closest MT unit postcode for each LSOA. Dictionary.

    mt_transfer_time:
        Time (minutes) for closest IVT to MT transfer. Dictionary.

    mt_transfer_unit:
        Closest MT unit for each IVT unit.  Dictionary.

    process_time_ambulance_response:
        Min/max of time from 999 call to ambulance arrival (tuple of integers)

    run_duration:
        Simulation run time (minutes, including warm-up). Integer.

    total_admissions:
        Total yearly admissions (obtained from LSOA admissions). Float.

    warm_up:
        Simulation run time (minutes) before audit starts.


    Methods
    -------
    load_data:
        Loads data to be used

    _load_hospitals:
        Loads data on the requested stroke teams.

    _load_admissions:
        Loads admissions data for the requested stroke teams and
        the LSOAs in their catchment areas.

    _load_lsoa_travel:
        Loads data on travel times from each LSOA to its nearest
        stroke units offering IVT and MT.

    _load_stroke_unit_travel:
        Loads data on travel times between stroke units.
    """

    def __init__(
            self,
            *initial_data,
            **kwargs
            ):
        """
        Constructor method for model parameters

        """
        # ----- Directory setup -----
        # Name that will also be used for output directory:
        self.name = 'scenario'
        # Load existing data from this dir:
        self.load_dir = None
        # Load existing parameters from this file:
        self.scenario_yml = 'scenario.yml'
        # Whether or not to make a new directory
        # (if not, risk overwriting existing files):
        self.make_new_scenario_dir = True

        # ----- Geography -----
        # Which LSOAs, stroke units, and regions will we use?
        self.limit_to_england = False
        self.limit_to_wales = False
        self.lsoa_catchment_type = 'nearest'

        # ----- Simpy parameters -----
        # The following batch of parameters are not called anywhere
        # during Scenario() or associated setup classes.
        # They will be called during the Patient, Pathway, and Model
        # classes.
        self.run_duration = 365  # Days
        self.warm_up = 50
        # What are the chances of treatment?
        self.probability_ivt = 1.0
        self.probability_mt = 1.0
        # Set process times.
        # Each tuple contains (minimum time, maximum time).
        # When both values are the same, all generated times
        # are that same value with no variation.
        self.process_time_call_ambulance = (30, 30)
        self.process_time_ambulance_response = (30, 30)
        self.process_ambulance_on_scene_duration = (20, 20)
        self.process_time_arrival_to_needle = (30, 30)
        self.process_time_arrival_to_puncture = (45, 45)
        self.transfer_time_delay = 30
        self.process_time_transfer_arrival_to_puncture = (60, 60)

        # ----- Overwrite default values -----
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # ----- Load helper classes -----
        # If no setup object was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()
        # Create calculations object:
        self.calculations = Calculations({'setup': self.setup})

        # ----- Load Scenario from files -----
        self.load_scenario_from_files()

        # ----- Make output directories -----
        self.check_output_directories()

        # Convert run duration to minutes
        self.run_duration *= 1440

    def load_scenario_from_files(self):
        if self.load_dir is None:
            # Update path to scenario files:
            self.setup.dir_scenario = self.name
            # self.setup.dir_scenario = os.path.join(
            #     self.setup.dir_output_all_scenarios,
            #     self.name
            # )
        else:
            self.name = os.path.split(self.load_dir)[-1]
            self.setup.dir_scenario = self.load_dir
            # # Update path to scenario files:
            # self.setup.dir_scenario = os.path.join(
            #     self.setup.dir_output_all_scenarios,
            #     self.name
            # )
            # Import the kwargs from provided yml file:
            path_to_scenario_yml = os.path.join(
                self.setup.dir_scenario, self.scenario_yml)
            with open(path_to_scenario_yml, 'r') as f:
                scenario_vars_imported = yaml.safe_load(f)
            # Save the imported kwargs to self:
            for key, val in scenario_vars_imported.items():
                setattr(self, key, val)

            # Create a new pathway/ dir for outputs.
            self.setup.dir_output_pathway = os.path.join(
                self.setup.dir_scenario,
                self.setup.name_dir_output_pathway
            )
            try:
                os.mkdir(self.setup.dir_output_pathway)
            except FileExistsError:
                # The directory already exists.
                pass

            # Load in any data files that are present:
            self.import_dataframes_from_file()

    def check_output_directories(self):
        # Pass in a keyword for whether to make a new directory
        # or keep the given name and overwrite anything in there.
        if self.make_new_scenario_dir is True:
            try:
                os.mkdir(self.setup.dir_scenario)
                rename_scenario = False
            except FileExistsError:
                # The directory already exists.
                rename_scenario = True
            if rename_scenario:
                # Rename the scenario and create a new directory
                # in the new name.
                # Return here because the output dir will be changed if
                # a dir with the same name already exists.
                self.setup.dir_scenario = self.setup.create_output_dir(
                    self.setup.dir_scenario)
                # os.mkdir(self.setup.dir_scenario)
                self.setup.update_scenario_list()
            else:
                # Nothing to do here.
                pass
        else:
            # Use the existing output directory.
            pass

        # Create a new pathway/ dir for outputs.
        self.setup.dir_output_pathway = os.path.join(
            self.setup.dir_scenario,
            self.setup.name_dir_output_pathway
        )
        try:
            os.mkdir(self.setup.dir_output_pathway)
        except FileExistsError:
            # The directory already exists.
            pass

    def import_dataframes_from_file(self):
        """
        Load dataframes from file.

        TO DO - write me.

        don't want to store a .yml with a bunch of DataFrames in it.
        Plan:
        Load Setup vars,
        look for each file in turn,
        setattr(name here, data).
        If file doesn't exist, pass -
        might want to load Scenario with only input files e.g.

        # Expect the following to be set before running this:
        # + name  # (from directory name?)
        # + setup
        # + units
        """
        # These can be loaded from file if they exist.
        # Set up csv contents (header and index columns):
        data_dicts = {
            'df_selected_regions': dict(
                csv_header=0,
                csv_index=None,
                file='file_selected_regions',
                func=self.set_model_areas
            ),
            'df_selected_units': dict(
                csv_header=0,
                csv_index=None,
                file='file_selected_units',
                func=self.set_unit_services
            ),
            'df_selected_transfer_units': dict(
                csv_header=0,
                csv_index=0,
                file='file_selected_transfer_units',
                func=self.set_transfer_units
            ),
            'df_selected_lsoa_catchment_nearest': dict(
                csv_header=0,
                csv_index=0,
                file='file_selected_lsoa_catchment_nearest',
                func=self.set_lsoa_catchment_nearest
            ),
            'df_selected_lsoa_catchment_island': dict(
                csv_header=0,
                csv_index=0,
                file='file_selected_lsoa_catchment_island',
                func=self.set_lsoa_catchment_island
            ),
            'df_selected_lsoa_admissions': dict(
                csv_header=0,
                csv_index=[0, 1],
                file='file_selected_lsoa_admissions',
                func=self.set_admissions
            ),
        }
        # Expect to find each file in either the given dir_scenario
        # or a subdirectory of it.
        # Look first in the pathway/ subdirectory, then in the main
        # directory, then give up.
        for key, data_dict in data_dicts.items():
            # Import from pathway subdirectory.
            path_to_file = os.path.join(
                self.setup.dir_output_pathway,
                getattr(self.setup, data_dict['file'])
            )
            if os.path.exists(path_to_file):
                pass
            else:
                # Import from main directory.
                path_to_file = os.path.join(
                    self.setup.dir_scenario,
                    getattr(self.setup, data_dict['file'])
                )
                if os.path.exists(path_to_file):
                    pass
                else:
                    # Give up.
                    path_to_file = None
            if path_to_file is None:
                # Don't attempt to load.
                pass
            else:
                df = pd.read_csv(
                    path_to_file,
                    index_col=data_dict['csv_index'],
                    header=data_dict['csv_header']
                )
                # Run the function to set this:
                data_dict['func'](df)

        # Set df_lsoa to whichever catchment type we selected.
        try:
            self.set_lsoa_catchment_type(self.lsoa_catchment_type)
        except AttributeError:
            # The necessary LSOA data doesn't exist yet.
            pass
        if hasattr(self, 'df_lsoa'):
            # Load in the travel time and unit dicts.
            self.create_lsoa_travel_dicts()

        # If admissions DataFrame exists, check that the related
        # parameters exist too.
        if hasattr(self, 'df_selected_lsoa_admissions'):
            params_bool = (hasattr(self, 'total_admissions') &
                           hasattr(self, 'inter_arrival_time'))
            if params_bool:
                # Don't need to do anything.
                pass
            else:
                # Create those two parameters.
                self.process_admissions(self.df_selected_lsoa_admissions)

    def save_to_file(self):
        """Save the variable dict as a .yml file."""
        scenario_vars = vars(self)

        # Only keep a selection of params:
        types_to_keep = [float, int, str]

        vars_to_save = {}
        for key, val in scenario_vars.items():
            if any([isinstance(val, t) for t in types_to_keep]):
                if isinstance(val, np.float64):
                    val = val.tolist()
                vars_to_save[key] = val

        dir_output = self.setup.dir_output_pathway
        file_output = 'scenario_output.yml'
        file_setup_vars = os.path.join(dir_output, file_output)

        with open(file_setup_vars, 'w') as f:
            yaml.dump(vars_to_save, f)

    # ###############################
    # ##### MAIN SETUP FUNCTION #####
    # ###############################
    def process_scenario(self):
        """
        Some of the attributes might already exist depending on how
        the data has been loaded in, so at each step check if it
        already exists or not.
        """
        if hasattr(self, 'df_selected_regions'):
            pass
        else:
            regions = self.get_model_areas()
            # ... probably want some user interaction here...
            self.set_model_areas(regions)

        if hasattr(self, 'df_selected_units'):
            pass
        else:
            units = self.get_unit_services()
            # ... probably want some user interaction here...
            self.set_unit_services(units)

        if hasattr(self, 'df_selected_transfer_units'):
            pass
        else:
            transfer = self.get_transfer_units()
            self.set_transfer_units(transfer)

        if hasattr(self, 'df_lsoa'):
            pass
        else:
            self.set_lsoa_catchment_type(self.lsoa_catchment_type)

        self.create_lsoa_travel_dicts()
        # Stores:
        # + self.lsoa_ivt_travel_time
        # + self.lsoa_ivt_unit

        if hasattr(self, 'df_selected_lsoa_admissions'):
            pass
        else:
            admissions = self.get_admissions()
            self.set_admissions(admissions)

        # self.make_dicts_for_pathway()

        # Save any parameters we've just calculated to .yml.
        self.save_to_file()

    def reset_scenario_data(self):
        # Delete the DataFrames.
        vars_to_delete = [
            'df_selected_regions',
            'df_selected_units',
            'df_selected_transfer_units',
            'df_selected_lsoa_catchment_nearest',
            'df_selected_lsoa_catchment_island',
            'df_selected_lsoa_admissions'
        ]
        for v in vars_to_delete:
            delattr(self, v)

    # ##########################
    # ##### AREAS TO MODEL #####
    # ##########################
    def get_model_areas(self):
        try:
            df = self.df_selected_regions
        except AttributeError:
            # TO DO - replace with relative import e.g. from stroke_outcome:
            # from importlib_resources import files  # For defining paths.
            # filename = files('stroke_outcome.data').joinpath(
            #     'mrs_dist_probs_cumsum.csv')
            # Load and parse area data
            dir_input = self.setup.dir_reference_data
            file_input = self.setup.file_input_regions
            path_to_file = os.path.join(dir_input, file_input)
            df = pd.read_csv(path_to_file)

            # Add a "selected" column for user input.
            df['selected'] = 0
        return df

    def set_model_areas(self, df):
        # TO DO - run sanity checks

        # TO DO - add an option to load this from a custom file.
        self.df_selected_regions = df

        # Save output to output folder.
        dir_output = self.setup.dir_output_pathway
        file_name = self.setup.file_selected_regions
        path_to_file = os.path.join(dir_output, file_name)
        df.to_csv(path_to_file, index=False)

        try:
            df_units = self.df_selected_units
        except AttributeError:
            # Automatically create units based on these areas.
            df_units = self.get_unit_services()
            self.set_unit_services(df_units)

    # #########################
    # ##### UNIT SERVICES #####
    # #########################
    def get_unit_services(self):
        try:
            df = self.df_selected_units
        except AttributeError:
            # Load and parse unit data
            dir_input = self.setup.dir_reference_data
            file_input = self.setup.file_input_unit_services
            path_to_file = os.path.join(dir_input, file_input)
            df = pd.read_csv(path_to_file)

            # Drop the LSOA column.
            df = df.drop('lsoa', axis='columns')
            # Add a "selected" column for user input.
            df['selected'] = 0

            try:
                # Load in the selected areas.
                df_areas = self.df_selected_regions
                # Which areas were selected?
                selected = df_areas['region'][df_areas['selected'] == 1]
                # Shouldn't contain repeats or NaN, but just in case:
                selected = selected.dropna().unique()
                # Set "selected" to 1 for any unit in the
                # selected areas.
                mask = df['region'].isin(selected)
                # Also only select units offering IVT.
                # TO DO - might not always be IVT? -----------------------------
                mask = mask & (df['use_ivt'] == 1)
                df.loc[mask, 'selected'] = 1
            except AttributeError:
                # self.df_selected_regions has not yet been set.
                pass
        return df

    def set_unit_services(self, df):
        # TO DO - run sanity checks
        # TO DO - add an option to load this from a custom file.

        self.df_selected_units = df

        # Save output to output folder.
        dir_output = self.setup.dir_output_pathway
        file_name = self.setup.file_selected_units
        path_to_file = os.path.join(dir_output, file_name)
        df.to_csv(path_to_file, index=False)

        # Calculate transfer unit info:
        self.get_transfer_units()

    # ##########################
    # ##### TRANSFER UNITS #####
    # ##########################
    def get_transfer_units(self):
        """
        write me

        TO DO - just keep a copy of national transfer units with 'selected' column?
        """
        self.national_dict = self.calculations.load_data()#_find_national_mt_feeder_units()

        # Merge in transfer unit names.
        # Load and parse hospital transfer data
        dir_input = self.setup.dir_output_pathway
        file_input = self.setup.file_national_transfer_units
        path_to_file = os.path.join(dir_input, file_input)
        transfer = pd.read_csv(path_to_file)
        transfer = transfer.rename(columns={'from_postcode': 'postcode'})
        transfer = transfer.drop(['time_nearest_mt'], axis='columns')
        # Index: 'Postcode'
        # Columns: 'name_nearest_mt'

        units = self.df_selected_units
        # Index: 'Postcode'
        # Columns: names, services, regions etc. ...

        # Limit to selected stroke units.
        selected_units = units['postcode'][units['selected'] == 1]
        mask = transfer['postcode'].isin(selected_units)
        transfer = transfer[mask]
        transfer = transfer.set_index('postcode')

        return transfer

    def set_transfer_units(self, transfer):
        self.df_selected_transfer_units = transfer

        # Save output to output folder.
        dir_output = self.setup.dir_output_pathway
        file_name = self.setup.file_selected_transfer_units
        path_to_file = os.path.join(dir_output, file_name)
        transfer.to_csv(path_to_file)

    # #########################
    # ##### SELECTED LSOA #####
    # #########################
    def find_lsoa_catchment_nearest(self):
        """
        TO DO - write me
        """
        (df_results, region_codes_containing_lsoa,
         region_codes_containing_units, units_catching_lsoa) = (
            self.calculations.find_lsoa_catchment_nearest(
                self.df_selected_units,
                self,
                treatment='ivt',
            ))

        self.set_lsoa_catchment_nearest(df_results)

        # Update regions data with whether contain LSOA...
        df_regions = self.df_selected_regions
        df_regions['contains_selected_lsoa'] = 0
        mask = df_regions['region_code'].isin(region_codes_containing_lsoa)
        df_regions.loc[mask, 'contains_selected_lsoa'] = 1
        # ... and units.
        df_regions['contains_unit_catching_lsoa'] = 0
        mask = df_regions['region_code'].isin(region_codes_containing_units)
        df_regions.loc[mask, 'contains_unit_catching_lsoa'] = 1
        # Save to self and to file:
        self.set_model_areas(df_regions)

        # Update units data with whether catch LSOA in selected regions.
        df_units = self.df_selected_units
        df_units['catches_lsoa_in_selected_region'] = 0
        mask = df_units['postcode'].isin(units_catching_lsoa)
        df_units.loc[mask, 'catches_lsoa_in_selected_region'] = 1
        self.set_unit_services(df_units)

    def set_lsoa_catchment_nearest(self, df_results):
        """
        Assume that if this is run directly instead of through
        find_lsoa_catchment_nearest that the other dataframes have
        already been updated - have columns for:
        + contains_selected_lsoa,
        + contains_unit_catching_lsoa,
        + catches_lsoa_in_selected_region
        """
        # Save output to output folder.
        dir_output = self.setup.dir_output_pathway
        file_name = self.setup.file_selected_lsoa_catchment_nearest
        path_to_file = os.path.join(dir_output, file_name)
        df_results.to_csv(path_to_file)

        # Save to self.
        self.df_selected_lsoa_catchment_nearest = df_results

    def find_lsoa_catchment_island(self):
        """
        TO DO - write me
        """
        df_results = self.calculations.find_lsoa_catchment_island(
            self.df_selected_units,
            self,
            treatment='ivt',
        )

        self.set_lsoa_catchment_island(df_results)

        # Don't need the following columns for island mode,
        # but Combine and Map classes are expecting to find them.
        # Update regions data with whether contain LSOA.
        df_regions = self.df_selected_regions
        save_file = False
        if 'contains_selected_lsoa' in df_regions.columns:
            # Don't overwrite the existing data.
            pass
        else:
            df_regions['contains_selected_lsoa'] = pd.NA
            save_file = True
        if 'contains_unit_catching_lsoa' in df_regions.columns:
            # Don't overwrite the existing data.
            pass
        else:
            df_regions['contains_unit_catching_lsoa'] = pd.NA
            save_file = True
        if save_file:
            self.set_model_areas(df_regions)

        # Update units data with whether catch LSOA in selected regions.
        df_units = self.df_selected_units
        if 'catches_lsoa_in_selected_region' in df_units.columns:
            # Don't overwrite the existing data.
            pass
        else:
            df_units['catches_lsoa_in_selected_region'] = pd.NA
            self.set_unit_services(df_units)

    def set_lsoa_catchment_island(self, df_results):
        """
        Assume that if this is run directly instead of through
        find_lsoa_catchment_nearest that the other dataframes have
        already been updated - have columns for:
        + contains_selected_lsoa,
        + contains_unit_catching_lsoa,
        + catches_lsoa_in_selected_region
        """
        # Save output to output folder.
        dir_output = self.setup.dir_output_pathway
        file_name = self.setup.file_selected_lsoa_catchment_island
        path_to_file = os.path.join(dir_output, file_name)
        df_results.to_csv(path_to_file)

        # Save to self.
        self.df_selected_lsoa_catchment_island = df_results

    def set_lsoa_catchment_type(self, lsoa_catchment_type):
        """
        TO DO - write me
        """
        # TO DO - run sanity checks

        if lsoa_catchment_type == 'island':
            try:
                self.df_lsoa = self.df_selected_lsoa_catchment_island
            except AttributeError:
                # If that data doesn't exist yet, make it now:
                self.find_lsoa_catchment_island()
                self.df_lsoa = self.df_selected_lsoa_catchment_island
            # Which LSOA selection file should be used?
            self.setup.file_selected_lsoas = (
                self.setup.file_selected_lsoa_catchment_island)
        else:
            try:
                self.df_lsoa = self.df_selected_lsoa_catchment_nearest
            except AttributeError:
                # If that data doesn't exist yet, make it now:
                self.find_lsoa_catchment_nearest()
                self.df_lsoa = self.df_selected_lsoa_catchment_nearest
            # Which LSOA selection file should be used?
            self.setup.file_selected_lsoas = (
                self.setup.file_selected_lsoa_catchment_nearest)

        self.lsoa_catchment_type = lsoa_catchment_type

    def create_lsoa_travel_dicts(self):
        """
        Convert LSOA travel time dataframe into separate dicts.
        """
        # Now create dictionaries of the LSOA travel times.
        df_travel = self.df_lsoa
        # TO DO - make this updateable for MT, MSU -------------------------------------
        # ..?
        treatments = ['ivt']

        # Separate out the columns and store in self:
        for treatment in treatments:
            travel_key = f'lsoa_{treatment}_travel_time'
            travel_val = df_travel[f'time_nearest_{treatment}']
            setattr(self, travel_key, travel_val)

            unit_key = f'lsoa_{treatment}_unit'
            unit_val = df_travel[f'postcode_nearest_{treatment}']
            setattr(self, unit_key, unit_val)

    # ######################
    # ##### ADMISSIONS #####
    # ######################
    def get_admissions(self):
        """
        Load admission data on the selected stroke teams.

        If no units are specified but "limit_to_england" is True,
        then only English stroke units are kept. If no units are
        specified and "limit_to_england" is False, then all stroke
        units are kept.

        Stores
        ------

        total_admissions:
            float. Total admissions in a year across selected
            stroke units.

        lsoa_relative_frequency:
            np.array. Relative frequency of each considered LSOA
            in the admissions data. Same order as self.lsoa_names.

        lsoa_names:
            np.array. Names of all LSOAs considered.
            Same order as lsoa_relative_frequency.

        inter_arrival_time:
            float. Average time between admissions in the
            considered stroke teams.
        """
        # Load and parse admissions data
        dir_input = self.setup.dir_reference_data
        file_input = self.setup.file_input_admissions
        path_to_file = os.path.join(dir_input, file_input)
        admissions = pd.read_csv(path_to_file)

        # Keep only these LSOAs in the admissions data:
        admissions = pd.merge(
            left=self.df_lsoa,
            right=admissions,
            left_on='lsoa',
            right_on='area',
            how='inner'
        )

        self.process_admissions(admissions)

        # Relative frequency of admissions across a year:
        admissions['relative_frequency'] = (
            admissions['admissions'] / self.total_admissions)
        # Set index to both LSOA name and code so that both follow
        # through to all of the results data.
        admissions = admissions.set_index(['area', 'lsoa_code'])

        # TO DO - need a separate admissions list for each LSOA catchment type.
        # Add it into the existing LSOA file.

        return admissions

    def set_admissions(self, admissions):
        # Save to self:
        self.df_selected_lsoa_admissions = admissions

        # Save output to output folder.
        dir_output = self.setup.dir_output_pathway
        file_name = self.setup.file_selected_lsoa_admissions
        path_to_file = os.path.join(dir_output, file_name)
        admissions.to_csv(path_to_file)

    def process_admissions(self, admissions):
        """
        Get some stats from the existing admissions DataFrame.
        """
        # Total admissions across these hospitals in a year:
        # Keep .tolist() to convert from np.float64 to float.
        self.total_admissions = np.round(
            admissions["admissions"].sum(), 0).tolist()
        # Average time between admissions to these hospitals in a year:
        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions

    # def make_dicts_for_pathway(self):
    #     self.lsoa_ivt_unit = self.df_lsoa['postcode_nearest_ivt'].to_dict()
    #     self.lsoa_ivt_travel_time = self.df_lsoa['time_nearest_ivt'].to_dict()