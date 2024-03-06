"""
Scenario class with global parameters for the pathway model.

TO DO -----------------------------------------------------------------------------
- write this docstring
- only save inputs to Scenario() for init?

        Assume that if this is run directly instead of through
        find_lsoa_catchment_nearest that the other dataframes have
        already been updated - have columns for:
        + contains_selected_lsoa,
        + contains_unit_catching_lsoa,
        + catches_lsoa_in_selected_region

        # TO DO - need to run the checks for LSOA catchment in other regions, by other units
        # even when they've been imported from file.

Usually the scenario.yml file should contain:

'name'  # this scenario
# Geography setup: for LSOA, region, units...
'limit_to_england',  # bool
'limit_to_wales',  # bool
'lsoa_catchment_type',  # 'island' or 'nearest'


TO DO - should this all be functions too? Not a class?
No, too many files and stuff in self.
"""
import numpy as np
import pandas as pd
import os
import yaml

from classes.calculations import Calculations
from classes.setup import Setup


class Scenario(object):  # TO DO - rename this - e.g. geography wrangling, catchment wrangling?
    """
    Global variables for model.

    class Scenario():

    Attributes
    ----------

    Methods
    -------

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
        if self.load_dir is None:
            # Update path to scenario files:
            self.setup.dir_scenario = self.name
            # self.setup.dir_scenario = os.path.join(
            #     self.setup.dir_output_all_scenarios,
            #     self.name
            # )
        else:
            self.load_scenario_from_files()

        # ----- Make output directories -----
        self.check_output_directories()

    def load_scenario_from_files(self):
        """
        """
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
        """
        """
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
            # No point loading these two from file?
            # 'df_selected_transfer_units': dict(
            #     csv_header=0,
            #     csv_index=0,
            #     file='file_selected_transfer_units',
            #     func=self.set_transfer_units
            # ),
            # 'df_selected_lsoa': dict(
            #     csv_header=0,
            #     csv_index=0,
            #     file='file_selected_lsoa',
            #     func=self.set_lsoa
            # )
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

        file_setup_vars = os.path.join(self.setup.dir_output_pathway,
                                       'scenario_output.yml')

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

        if hasattr(self, 'df_selected_lsoa'):
            pass
        else:
            self.calculate_lsoa_catchment()
            try:
                self.df_selected_lsoa['admissions']
            except KeyError:
                admissions = self.load_admissions()
                self.set_admissions(admissions)

        # Save any parameters we've just calculated to .yml.
        self.save_to_file()

    def reset_scenario_data(self):
        # Delete the DataFrames.
        vars_to_delete = [
            'df_selected_regions',
            'df_selected_units',
            'df_selected_transfer_units',
            'df_selected_lsoa',
        ]
        for v in vars_to_delete:
            delattr(self, v)

    # ##########################
    # ##### AREAS TO MODEL #####
    # ##########################
    def get_model_areas(self):
        """
        write me
        """
        try:
            df = self.df_selected_regions
        except AttributeError:
            # TO DO - replace with relative import e.g. from stroke_outcome:
            # from importlib_resources import files  # For defining paths.
            # filename = files('stroke_outcome.data').joinpath(
            #     'mrs_dist_probs_cumsum.csv')
            # Load and parse area data
            path_to_file = os.path.join(self.setup.dir_reference_data,
                                        self.setup.file_input_regions)
            df = pd.read_csv(path_to_file)

            # Add a "selected" column for user input.
            df['selected'] = 0
        return df

    def set_model_areas(self, df):
        """
        write me
        """
        # TO DO - run sanity checks

        self.df_selected_regions = df

        # Save output to output folder.
        path_to_file = os.path.join(self.setup.dir_output_pathway,
                                    self.setup.file_selected_regions)
        df.to_csv(path_to_file, index=False)

    # #########################
    # ##### UNIT SERVICES #####
    # #########################
    def get_unit_services(self):
        """
        write me
        """
        try:
            df = self.df_selected_units
        except AttributeError:
            # TO DO - replace with relative import
            # Load and parse unit data
            path_to_file = os.path.join(self.setup.dir_reference_data,
                                        self.setup.file_input_unit_services)
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
        """
        write me
        """
        # TO DO - run sanity checks
        # TO DO - add an option to load this from a custom file.

        self.df_selected_units = df

        # Save output to output folder.
        path_to_file = os.path.join(self.setup.dir_output_pathway,
                                    self.setup.file_selected_units)
        df.to_csv(path_to_file, index=False)

    # ##########################
    # ##### TRANSFER UNITS #####
    # ##########################
    def get_transfer_units(self):
        """
        write me

        If exists, don't load?
        """
        # Find which IVT units are feeders to each MT unit:
        transfer = self.calculations.find_national_mt_feeder_units()
        transfer = transfer.rename(columns={'from_postcode': 'postcode'})
        # transfer = transfer.drop(['time_nearest_mt'], axis='columns')
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
        """
        write me
        """
        self.df_selected_transfer_units = transfer

        # Save output to output folder.
        path_to_file = os.path.join(self.setup.dir_output_pathway,
                                    self.setup.file_selected_transfer_units)
        transfer.to_csv(path_to_file)

    # #########################
    # ##### SELECTED LSOA #####
    # #########################
    def calculate_lsoa_catchment(self):
        """
        TO DO - write me
        """
        # Regions data:
        df_regions = self.df_selected_regions
        mask = df_regions['selected'] == 1
        regions_selected = sorted(set(list(
            df_regions.loc[mask]['region_code'])))

        if self.lsoa_catchment_type == 'island':
            # Find list of selected regions:
            regions_to_limit = regions_selected
        else:
            regions_to_limit = []

        # For all LSOA:
        df_catchment = self.calculations.find_lsoa_catchment()
        # Limit to the useful LSOA:
        df_catchment = (
            self.calculations.limit_lsoa_catchment_to_selected_units(
                df_catchment,
                regions_selected,
                regions_to_limit=regions_to_limit,
                limit_to_england=self.limit_to_england,
                limit_to_wales=self.limit_to_wales
            ))

        # Units data for region matching:
        df_units = self.df_selected_units

        # Find which regions and units use these LSOA:
        tup = self.calculations.find_catchment_info_regions_and_units(
                df_catchment, df_units)
        # tup contains lists of:
        # + region codes containing lsoa
        # + units catching lsoa
        # + region codes containing units

        # Drop region columns from df_catchment:
        cols_to_drop = ['region', 'region_code', 'region_type']
        df_catchment = df_catchment.drop(cols_to_drop, axis='columns')

        self.update_data_with_lsoa_catchment(df_catchment, *tup)

    def update_data_with_lsoa_catchment(
            self,
            df_catchment,
            region_codes_containing_lsoa,
            units_catching_lsoa,
            region_codes_containing_units
            ):
        """
        write me
        """
        # ----- LSOA data -----
        # Save to self and to file:
        self.set_lsoa(df_catchment)

        # ----- Units -----
        df_units = self.df_selected_units
        # Update units data with whether catch LSOA in selected regions.
        df_units['catches_lsoa_in_selected_region'] = 0
        mask = df_units['postcode'].isin(units_catching_lsoa)
        df_units.loc[mask, 'catches_lsoa_in_selected_region'] = 1
        # Save to self and to file:
        self.set_unit_services(df_units)

        # ----- Regions -----
        df_regions = self.df_selected_regions
        # Update regions data with whether contain LSOA...
        df_regions['contains_selected_lsoa'] = 0
        mask = df_regions['region_code'].isin(region_codes_containing_lsoa)
        df_regions.loc[mask, 'contains_selected_lsoa'] = 1
        # ... and units.
        df_regions['contains_unit_catching_lsoa'] = 0
        mask = df_regions['region_code'].isin(region_codes_containing_units)
        df_regions.loc[mask, 'contains_unit_catching_lsoa'] = 1
        # Save to self and to file:
        self.set_model_areas(df_regions)

    def set_lsoa(self, df_results):
        """
        write me
        """
        # TO DO -
        # if save, then if setup exists, then... ----------------------------------
        # Save output to output folder.
        path_to_file = os.path.join(
            self.setup.dir_output_pathway,
            self.setup.file_selected_lsoa
            )
        df_results.to_csv(path_to_file)

        # Save to self.
        self.df_selected_lsoa = df_results

    # ######################
    # ##### ADMISSIONS #####
    # ######################
    def load_admissions(self):
        """
        Load admission data on the selected stroke teams.

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
        # TO DO - replace with relative import
        # Load and parse admissions data
        path_to_file = os.path.join(self.setup.dir_reference_data,
                                    self.setup.file_input_admissions)
        admissions = pd.read_csv(path_to_file)
        return admissions

    def match_admissions_to_selected_lsoa(self, admissions):
        """
        write me
        """
        # Keep only these LSOAs in the admissions data:
        admissions = pd.merge(left=self.df_lsoa, right=admissions,
                              left_on='lsoa', right_on='area', how='left')

        # Total admissions across these hospitals in a year:
        # Keep .tolist() to convert from np.float64 to float.
        total_admissions = np.round(
            admissions["admissions"].sum(), 0).tolist()

        # Relative frequency of admissions across a year:
        admissions['relative_frequency'] = (
            admissions['admissions'] / total_admissions)
        # Set index to both LSOA name and code so that both follow
        # through to all of the results data.
        admissions = admissions.set_index(['area', 'lsoa_code'])
        return admissions

    def set_admissions(self, admissions):
        """
        write me
        """
        # Save to self and to file:
        self.set_lsoa(admissions)
