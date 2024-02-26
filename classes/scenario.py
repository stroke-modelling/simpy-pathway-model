"""
Scenario class with global parameters for the pathway model.

TO DO -----------------------------------------------------------------------------
- write this docstring
- load national data from file
- save vars to yml - what about dataframes? (repr?)
- only save inputs to Scenario() for init?
"""
import numpy as np
import pandas as pd
import os
import yaml

from classes.units import Units
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

    def __init__(self, *initial_data, **kwargs):
        """Constructor method for model parameters"""
        # Name that will also be used for output directory:
        self.name = 'scenario'

        # ----- Geography -----
        # Which LSOAs, stroke units, and regions will we use?
        self.limit_to_england = False
        self.limit_to_wales = False
        self.lsoa_catchment_type = 'nearest'

        # ----- Simpy parameters: -----
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

        # If no setup was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()

        # TO DO - overhaul dir creation and lookup bits --------------------------
        # And create an output directory for this Scenario:
        dir_output = self.name
        # Return here because the output dir will be changed if
        # a dir with the same name already exists.
        dir_output = self.setup.create_output_dir(dir_output)

        # Convert run duration to minutes
        self.run_duration *= 1440

        self.units = Units({'setup': self.setup})

    def import_from_file(self, dir_scenario, scenario_yml='scenario.yml'):
        """
        Import a .yml file and overwrite attributes here.

        TO DO - don't want to store a .yml with a bunch of DataFrames in it.
        Plan:
        Load Setup vars,
        look for each file in turn,
        find file directory too, build path,
        import data from file
        setattr(name here, data).
        So need a list of var names here and the matching file name from setup.
        If file doesn't exist, pass - might want to load Scenario with only input files e.g.

        # Expect the following to be set before running this:
        # + name  # (from directory name?)
        # + setup
        # + units
        """

        # Import the following from a scenario.yml file:
        scenario_vars_keys = [
            # Geography setup: for LSOA, region, units...
            'limit_to_england',  # bool
            'limit_to_wales',  # bool
            'lsoa_catchment_type',  # 'island' or 'nearest'
            # TO DO - convert admissions stuff to DataFrame, save to file.
            'total_admissions',         # float
            'lsoa_relative_frequency',  # array
            'lsoa_names',               # array. Names, codes in same order as lsoa_relative frequency.
            'inter_arrival_time'        # float
            # Pathway parameters:
            'run_duration',
            'warm_up',
            'probability_ivt',
            'probability_mt',
            'process_time_call_ambulance',
            'process_time_ambulance_response',
            'process_ambulance_on_scene_duration',
            'process_time_arrival_to_needle',
            'process_time_arrival_to_puncture',
            'transfer_time_delay',
            'process_time_transfer_arrival_to_puncture',
            ]

        path_to_scenario_yml = os.path.join(dir_scenario, scenario_yml)
        with open(path_to_scenario_yml, 'r') as f:
            scenario_vars_imported = yaml.safe_load(f)

        for key, val in scenario_vars_imported.items():
            setattr(self, key, val)

        # These can be loaded from file if they exist:
        data_in_files = {
            'df_selected_regions': self.setup.file_selected_regions,
            'df_selected_units': self.setup.file_selected_units,
            'df_transfer_units': self.setup.selected_transfer_units,
            'df_lsoa_catchment_nearest': self.setup.selected_lsoa_catchment_nearest,
            'df_lsoa_catchment_island': self.setup.selected_lsoa_catchment_island,
        }
        # Expect to find each file in either the given dir_scenario
        # or a subdirectory of it.

        # Which file names are they stored in?
        # Which csv headers and indices do we need?

        # Set df_lsoa to whichever catchment type we selected.
        try:
            self.set_lsoa_catchment_type(self.lsoa_catchment_type)
        except AttributeError:
            # The necessary LSOA data doesn't exist yet.
            pass
        if hasattr(self, 'df_lsoa'):
            # Load in the travel time and unit dicts.
            self._create_lsoa_travel_dicts()

    # ###############################
    # ##### MAIN SETUP FUNCTION #####
    # ###############################
    def load_data(self):
        """
        TO DO - update me ------------------------------------------------------

        Load required data.

        Stores the following in the Globvars object:
        + hospitals
        + lsoa_names
        + total_admissions
        + lsoa_relative_frequency
        + inter_arrival_time
        + lsoa_ivt_travel_time
        + lsoa_ivt_unit
        + lsoa_mt_travel_time
        + lsoa_mt_unit
        + lsoa_msu_travel_time
        + lsoa_msu_unit
        + mt_transfer_time
        + mt_transfer_unit

        More details on each attribute are given in the docstrings
        of the methods that create them.
        """
        # Find which LSOAs are in these stroke teams' catchment areas:
        self._create_lsoa_travel_dicts()
        # Stores:
        # + self.lsoa_names
        #   --> saves to: file_selected_lsoas
        # + self.lsoa_ivt_travel_time
        # + self.lsoa_ivt_unit
        # + self.lsoa_mt_travel_time
        # + self.lsoa_mt_unit
        # + self.lsoa_msu_travel_time
        # + self.lsoa_msu_unit

        # Import admissions statistics for those hospitals:
        self._load_admissions()
        # Stores:
        # + self.total_admissions
        # + self.lsoa_relative_frequency
        # + self.inter_arrival_time

    # ##########################
    # ##### AREAS TO MODEL #####   --> should this move to Units()?
    # ##########################
    def get_model_areas(self):
        try:
            df = self.df_selected_regions
        except AttributeError:
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
        dir_output = self.setup.dir_output
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
    # ##### UNIT SERVICES #####   --> should this move to Units()?
    # #########################
    def get_unit_services(self):

        # TO DO - make this show less stuff to the user.
        # Remove region codes and stuff.

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
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_units
        path_to_file = os.path.join(dir_output, file_name)
        df.to_csv(path_to_file, index=False)

        # Calculate transfer unit info:
        self._load_transfer_units()

    def _load_transfer_units(self):
        """
        write me

        TO DO - just keep a copy of national transfer units with 'selected' column?
        """
        # Merge in transfer unit names.
        # Load and parse hospital transfer data
        dir_input = self.setup.dir_output
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

        self.df_transfer_units = transfer

        # Save output to output folder.
        dir_output = self.setup.dir_output
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
            self.units.find_lsoa_catchment_nearest(
                self.df_selected_units,
                self,
                treatment='ivt',
            ))

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_lsoa_catchment_nearest
        path_to_file = os.path.join(dir_output, file_name)
        df_results.to_csv(path_to_file)

        # Save to self.
        self.df_lsoa_catchment_nearest = df_results

        # Update regions data with whether contain LSOA...
        df_regions = self.df_selected_regions
        df_regions['contains_selected_lsoa'] = 0
        mask = df_regions['region_code'].isin(region_codes_containing_lsoa)
        df_regions.loc[mask, 'contains_selected_lsoa'] = 1
        # ... and units.
        df_regions['contains_unit_catching_lsoa'] = 0
        mask = df_regions['region_code'].isin(region_codes_containing_units)
        df_regions.loc[mask, 'contains_unit_catching_lsoa'] = 1
        # Save to self:
        self.set_model_areas(df_regions)

        # Update units data with whether catch LSOA in selected regions.
        df_units = self.df_selected_units
        df_units['catches_lsoa_in_selected_region'] = 0
        mask = df_units['postcode'].isin(units_catching_lsoa)
        df_units.loc[mask, 'catches_lsoa_in_selected_region'] = 1
        self.set_unit_services(df_units)

    def find_lsoa_catchment_island(self):
        """
        TO DO - write me
        """
        df_results = self.units.find_lsoa_catchment_island(
            self.df_selected_units,
            self,
            treatment='ivt',
        )

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_lsoa_catchment_island
        path_to_file = os.path.join(dir_output, file_name)
        df_results.to_csv(path_to_file)

        # Save to self.
        self.df_lsoa_catchment_island = df_results

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

    def set_lsoa_catchment_type(self, lsoa_catchment_type):
        """
        TO DO - write me
        """
        # TO DO - run sanity checks

        if lsoa_catchment_type == 'island':
            try:
                self.df_lsoa = self.df_lsoa_catchment_island
            except AttributeError:
                # If that data doesn't exist yet, make it now:
                self.find_lsoa_catchment_island()
                self.df_lsoa = self.df_lsoa_catchment_island
            # Which LSOA selection file should be used?
            self.setup.file_selected_lsoas = (
                self.setup.file_selected_lsoa_catchment_island)
        else:
            try:
                self.df_lsoa = self.df_lsoa_catchment_nearest
            except AttributeError:
                # If that data doesn't exist yet, make it now:
                self.find_lsoa_catchment_nearest()
                self.df_lsoa = self.df_lsoa_catchment_nearest
            # Which LSOA selection file should be used?
            self.setup.file_selected_lsoas = (
                self.setup.file_selected_lsoa_catchment_nearest)

        self.lsoa_catchment_type = lsoa_catchment_type

    def _create_lsoa_travel_dicts(self):
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

    def _load_admissions(self):
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

        # Process admissions.
        # Total admissions across these hospitals in a year:
        self.total_admissions = np.round(admissions["Admissions"].sum(), 0)
        # Average time between admissions to these hospitals in a year:
        self.inter_arrival_time = (365 * 24 * 60) / self.total_admissions

        # Relative frequency of admissions across a year:
        admissions['relative_frequency'] = (
            admissions['Admissions'] / self.total_admissions)
        # Save the LSOA names in the same order as the
        # LSOA relative frequency array.
        admissions = admissions.set_index(['area', 'lsoa_code'])
        self.lsoa_names = admissions.index.values
        self.lsoa_relative_frequency = (
            admissions['relative_frequency'].to_dict())

        # Save output to output folder.
        dir_output = self.setup.dir_output
        file_name = self.setup.file_selected_lsoa_admissions
        path_to_file = os.path.join(dir_output, file_name)
        admissions.to_csv(path_to_file)
