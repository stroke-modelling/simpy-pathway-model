"""
Draw some maps using output files.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).
"""

"""
NOTES
# Always load all data every time.
# Only load it once.
# Select different DataFrame columns for different plots.
# Separate dataframes for geography?

# for dir in output_dir_list:
# load in these files,
# combine into a single dataframe, 
# if only one dir then no problem.
# Same variables for everything, use dir name to mask data and plot different bits.

# Don't save a column named "geometry" or the load won't work.
# ... so then what?

# TO MERGE MULTIINDEX, have to know number of levels to add to the gdf data
# --> add levels then merge. Levels depends on whether using combo or single data.
# Make sure that dropping upper levels would recreate the single data version.
# TO DO.

Limit LSOAs to those whose nearest stroke units are in the list.

Example catchment areas:
The two right-hand units are in the selected regions
and the two units on the left are national units,
not modelled directly.

    ▓▓▓▓▓▓▓▓▓░░░░░░░░█████▒▒▒▒▒  <-- Drip-and-ship   +------------+
    ▏   *        o     o   *  ▕                      | * MT unit  |
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒  <-- Mothership      | o IVT unit |
                -->▏ ▕<--                             +------------+
                Difference

The catchment area boundaries are halfway between adjacent units.

In the mothership model, the right-hand MT unit's catchment area
covers the whole right half of the rectangle.
In the drip-and-ship model, some of that area is instead assigned
to the out-of-area IVT unit in the centre of the rectangle.

TO DO - currently brute forcing all of the dataframes to have mathcing
column index labels and levels and that, but should be able to function
it up. Automatically know what levels are already there and which need
adding.

TO DO - automatic repr() for making maps

TO DO - better colour assignment.
e.g. drip and ship catchment map, make all feeder units different shades
of the same colour for each MT unit.

"""
# TO DO
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.
from shapely.geometry import Polygon # For extent box.

from classes.setup import Setup

import classes.map_functions as maps  # for plotting.


class Map(object):
    """
    Combine files from multiple runs of the pathway.

    class Combine():

    TO DO - write me
    """
    def __init__(self, *initial_data, **kwargs):

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

    # ##########################
    # ##### FILE SELECTION #####
    # ##########################
    def load_run_data(self, load_list=[], dir_data=None):
        """
        Load in data specific to these runs.
        """
        # Setup for combined files:
        dicts_combo = {
            'df_units': {
                'file': self.setup.file_combined_selected_stroke_units,
                'header': [0, 1],
                'index_col': 0,
                },
            'df_transfer': {
                'file': self.setup.file_combined_selected_transfer_units,
                'header': [0, 1],
                'index_col': [0, 1],
                },
            'df_lsoa': {
                'file': self.setup.file_combined_selected_lsoas,
                'header': [0, 1],
                'index_col': 1,
                },
            'df_regions': {
                'file': self.setup.file_combined_selected_regions,
                'header': [0, 1],
                'index_col': 1,
                },
            'df_results_by_unit': {
                'file': (
                    self.setup.file_combined_results_summary_by_admitting_unit
                    ),
                'header': [0, 1, 2],
                'index_col': 0,
                },
            'df_results_by_lsoa': {
                'file': self.setup.file_combined_results_summary_by_lsoa,
                'header': [0, 1, 2],
                'index_col': 0,
                },
        }
        # Setup for individual run's files:
        dicts_single = {
            'df_units': {
                'file': self.setup.file_selected_stroke_units,
                'header': [0],
                'index_col': 0,
                },
            'df_transfer': {
                'file': self.setup.file_selected_transfer_units,
                'header': [0],
                'index_col': 0,
                },
            'df_lsoa': {
                'file': self.setup.file_selected_lsoas,
                'header': [0],
                'index_col': 1,
                },
            'df_regions': {
                'file': self.setup.file_selected_regions,
                'header': [0],
                'index_col': 1,
                },
            'df_results_by_unit': {
                'file': (
                    self.setup.file_results_summary_by_admitting_unit
                    ),
                'header': [0, 1],
                'index_col': 0,
                },
            'df_results_by_lsoa': {
                'file': self.setup.file_results_summary_by_lsoa,
                'header': [0, 1],
                'index_col': 0,
                },
        }
        if dir_data is None:
            # Use combined data files.
            dir_data = self.setup.dir_output_combined
            dicts_data = dicts_combo
            self.data_type = 'combined'
        else:
            # Use files for the selected scenario only.
            for d in self.setup.list_dir_output:
                end = os.path.split(d)[-1]
                if end == dir_data:
                    dir_data = d
            dicts_data = dicts_single
            self.data_type = 'single'

        if len(load_list) == 0:
            # Load everything.
            load_list = list(dicts_data.keys())

        for label in load_list:
            data_dict = dicts_data[label]
            # Make path to file:
            path_to_file = os.path.join(dir_data, data_dict['file'])
            try:
                # Specify header to import as a multiindex DataFrame.
                df = pd.read_csv(
                    path_to_file,
                    header=data_dict['header'],
                    index_col=data_dict['index_col']
                    )
                if ((label == 'df_transfer') & (self.data_type == 'single')):
                    # Add another column to the index.
                    iname = df.index.name
                    df = df.reset_index()
                    df = df.set_index([iname, 'name_nearest_MT'])
                # Save to self:
                setattr(self, label, df)
            except FileNotFoundError:
                # TO DO - proper error message
                raise FileNotFoundError(
                    f'Cannot import {label} from {data_dict["file"]}'
                    ) from None

    def import_geojson(self, region_type: 'str'):
        """
        Import a geojson file as GeoDataFrame.

        The crs (coordinate reference system) is set to British National
        Grid.

        Inputs
        ------
        setup       - Setup() object. Contains attributes for paths to the
                    data directory and the geojson file names.
        self.region_type - str. Lookup name for selecting a geojson file.
                    This should be one of the column names from the
                    various regions files.

        Returns
        -------
        gdf_boundaries - GeoDataFrame. One row per region shape in the
                        file. Expect columns for region name and geometry.
        """
        # Select geojson file based on input region type:
        geojson_file_dict = {
            'LSOA11NM': self.setup.file_geojson_lsoa,
            'SICBL22NM': self.setup.file_geojson_sibcl,
            # 'ICB22NM': self.setup.file_geojson_icb,
            'LHB20NM': self.setup.file_geojson_lhb,
        }

        # Import region file:
        dir_input = self.setup.dir_data_geojson
        file_input = geojson_file_dict[region_type]
        path_to_file = os.path.join(dir_input, file_input)
        gdf_boundaries = geopandas.read_file(path_to_file)

        if region_type.endswith('NM'):
            region_code = region_type[:-2] + 'CD'
        elif region_type.endswith('nm'):
            region_code = region_type[:-2] + 'cd'
        else:
            # This shouldn't happen.
            # TO DO - error handle for NMW?
            region_code = region_type[:-2] + 'CD'

        if region_type == 'LSOA11NM':
            # Set the index:
            gdf_boundaries = gdf_boundaries.set_index('LSOA11CD')

            # Only keep geometry data:
            geo_cols = [
                'BNG_E', 'BNG_N', 'LONG', 'LAT', 'GlobalID', 'geometry'
            ]
            # Don't keep LSOA11NM because that will be merged in later
            # from an LSOA dataframe.

        else:
            try:
                # Rename this column:
                gdf_boundaries = gdf_boundaries.rename(columns={
                        region_type: 'region',
                        region_code: 'region_code'
                    })
            except KeyError:
                # That column doesn't exist.
                # Try finding a column that has the same start and end
                # as requested:
                prefix = region_type[:3]
                suffix = region_type[-2:]
                success = False
                for column in gdf_boundaries.columns:
                    # Casefold turns all UPPER into lower case for the
                    # comparison.
                    match = (
                        (column[:3].casefold() == prefix.casefold()) &
                        (column[-2:].casefold() == suffix.casefold())
                        )
                    if match:
                        # Rename this column:
                        col_code = column[:-2] + region_code[-2:]
                        gdf_boundaries = gdf_boundaries.rename(columns={
                            column: 'region',
                            col_code: 'region_code'
                            })
                        success = True
                    else:
                        # TO DO - proper error here --------------------------------
                        pass

            # Set the index:
            gdf_boundaries = gdf_boundaries.set_index('region_code')

            # Only keep geometry data:
            geo_cols = [
                'region', 'BNG_E', 'BNG_N', 'LONG', 'LAT', 'GlobalID', 'geometry'
            ]

        gdf_boundaries = gdf_boundaries[geo_cols]

        # If crs is given in the file, geopandas automatically
        # pulls it through. Convert to National Grid coordinates:
        if gdf_boundaries.crs != 'EPSG:27700':
            gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
        return gdf_boundaries

    # def update_regions(self, region_type: str):
    #     self.region_type = region_type
    #     self.load_geometry_regions()

    # ##########################
    # ##### DATA WRANGLING #####
    # ##########################
    def process_data(self):
        self.load_geometry_lsoa()
        self.load_geometry_regions()
        self.load_geometry_stroke_units()
        self.load_geometry_transfer_units()

    def load_geometry_lsoa(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # ----- Gather data -----
        # Selected LSOA names, codes.
        # ['LSOA11NM', 'LSOA11CD', '{region}', 'postcode_nearest', 'Use']
        try:
            df_lsoa = self.df_lsoa
        except AttributeError:
            self.load_run_data(['df_lsoa'])
            df_lsoa = self.df_lsoa
        # Index column: LSOA11NM.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        # All LSOA shapes:
        gdf_boundaries_lsoa = self.import_geojson('LSOA11NM')
        # Index column: LSOA11CD.
        # Always has only one unnamed column index level.

        # Results by LSOA.
        results_exist = False
        try:
            # If the file wasn't loaded, this gives AttributeError:
            df_lsoa_results = self.df_results_by_lsoa
            # Index column: lsoa.
            # Expected column MultiIndex levels:
            #   - combined: ['scenario', 'property', 'subtype']
            #   - separate: ['property', 'subtype]
            results_exist = True
        except AttributeError:
            pass # TEMPORARY!!!
            # try:
            #     # self.load_run_data(['df_results_by_lsoa']) # TEMPORARY - TO DO - fix me
            #     # currently this will break again when sometimes want combo data, sometimes not -
            #     # need explicit call to load_data() or whatever.
            #     df_lsoa_results = self.df_results_by_lsoa
            #     results_exist = True
            # except FileNotFoundError:
            #     # Give up on loading this in.
            #     pass


        # ----- Prepare separate data -----
        # Set up column level info for the merged DataFrame.
        # The "combined" scenario Dataframe will have an extra
        # column level with the scenario name.
        if self.data_type == 'combined':
            # LSOA names:
            cols_lsoa = df_lsoa.columns
            df_lsoa_column_arr = np.array(
                [[n for n in c] for c in cols_lsoa])
            cols_df_lsoa = [
                df_lsoa_column_arr[:, 0],                    # scenario
                df_lsoa_column_arr[:, 1],                    # property
                [''] * len(cols_lsoa),                       # subtype
            ]
            # Geometry:
            cols_gdf_boundaries_lsoa = [
                ['any'] * len(gdf_boundaries_lsoa.columns),  # scenario
                gdf_boundaries_lsoa.columns,                 # property
                [''] * len(gdf_boundaries_lsoa.columns),     # subtype
            ]
            # Final data:
            col_level_names = ['scenario', 'property', 'subtype']
            col_geometry = ('any', 'geometry', '')
        else:
            # LSOA names:
            # cols_to_drop_lsoa = 'LSOA11CD'
            cols_lsoa = df_lsoa.columns#.drop(cols_to_drop_lsoa)
            cols_df_lsoa = [
                cols_lsoa,                                   # property
                [''] * len(cols_lsoa),                       # subtype
            ]
            # Geometry:
            cols_gdf_boundaries_lsoa = [
                gdf_boundaries_lsoa.columns,                 # property
                [''] * len(gdf_boundaries_lsoa.columns),     # subtype
            ]
            # Final data:
            col_level_names = ['property', 'subtype']
            col_geometry = ('geometry', '')

        # Drop columns that are duplicated across DataFrames.
        # df_lsoa = df_lsoa.drop(cols_to_drop_lsoa, axis='columns')

        # Make all data to be combined have the same column levels.
        # LSOA names:
        df_lsoa = pd.DataFrame(
            df_lsoa.values,
            index=df_lsoa.index,
            columns=cols_df_lsoa
        )

        # Geometry:
        gdf_boundaries_lsoa = pd.DataFrame(
            gdf_boundaries_lsoa.values,
            index=gdf_boundaries_lsoa.index,
            columns=cols_gdf_boundaries_lsoa
        )

        # Results:
        # This already has the right number of column levels.

        # ----- Create final data -----
        # Merge together all of the DataFrames.
        gdf_boundaries_lsoa = pd.merge(
            gdf_boundaries_lsoa, df_lsoa,
            left_index=True, right_index=True, how='right'
        )
        if results_exist:
            gdf_boundaries_lsoa = pd.merge(
                gdf_boundaries_lsoa, df_lsoa_results,
                left_index=True, right_index=True, how='left'
            )
        # Name the column levels:
        gdf_boundaries_lsoa.columns = (
            gdf_boundaries_lsoa.columns.set_names(col_level_names))

        if self.data_type == 'combined':
            # Sort the results by scenario (top column index):
            cols_scen = (
                gdf_boundaries_lsoa.columns.get_level_values('scenario'))
            cols_scen = sorted(list(set(cols_scen)))
            gdf_boundaries_lsoa = gdf_boundaries_lsoa[cols_scen]
        else:
            # Don't sort.
            pass

        # Convert to GeoDataFrame:
        gdf_boundaries_lsoa = geopandas.GeoDataFrame(
            gdf_boundaries_lsoa,
            geometry=col_geometry
            )

        # ----- Save to self -----
        self.gdf_boundaries_lsoa = gdf_boundaries_lsoa

    def load_geometry_regions(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.

        TO DO - load in both Wales and England if necessary.
        """
        # ----- Gather data -----
        # Selected regions names and usage.
        # ['{region type}', 'contains_selected_unit', 'contains_selected_lsoa']
        try:
            df_regions = self.df_regions
        except AttributeError:
            self.load_run_data(['df_regions'])
            df_regions = self.df_regions
        # Index column: 'region'.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        # All region polygons:
        gdf_boundaries_regions_e = self.import_geojson('SICBL22NM')
        gdf_boundaries_regions_w = self.import_geojson('LHB20NM')
        # Combine:
        gdf_boundaries_regions = pd.concat(
            (gdf_boundaries_regions_e, gdf_boundaries_regions_w),
            axis='rows'
        )
        # Index column: 'region'.
        # Always has only one unnamed column index level.

        # Drop columns in both DataFrames:
        gdf_boundaries_regions = gdf_boundaries_regions.drop(
            'region', axis='columns'
        )

        # ----- Prepare separate data -----
        # Set up column level info for the merged DataFrame.
        # The "combined" scenario Dataframe will have an extra
        # column level with the scenario name.
        if self.data_type == 'combined':
            # Geometry:
            cols_gdf_boundaries_regions = [
                ['any'] * len(gdf_boundaries_regions.columns),  # scenario
                gdf_boundaries_regions.columns,                 # property
            ]
            # Final data:
            col_level_names = ['scenario', 'property']
            col_geometry = ('any', 'geometry')
            col_colour = ('any', 'colour')
        else:
            # Geometry:
            cols_gdf_boundaries_regions = gdf_boundaries_regions.columns
            # Final data:
            col_level_names = ['property']
            col_geometry = 'geometry'
            col_colour = 'colour'

        # Geometry:
        # If necessary, add more column levels.
        gdf_boundaries_regions = pd.DataFrame(
            gdf_boundaries_regions.values,
            index=gdf_boundaries_regions.index,
            columns=cols_gdf_boundaries_regions
        )

        # ----- Create final data -----
        # Merge together the DataFrames.
        gdf_boundaries_regions = pd.merge(
            gdf_boundaries_regions, df_regions,
            left_index=True, right_index=True, how='right'
        )

        # Name the column levels:
        gdf_boundaries_regions.columns = (
            gdf_boundaries_regions.columns.set_names(col_level_names))

        if self.data_type == 'combined':
            # Sort the results by scenario (top column index):
            cols_scen = (
                gdf_boundaries_regions.columns.get_level_values('scenario'))
            cols_scen = sorted(list(set(cols_scen)))
            gdf_boundaries_regions = gdf_boundaries_regions[cols_scen]
        else:
            # Don't sort.
            pass

        # Convert to GeoDataFrame:
        gdf_boundaries_regions = geopandas.GeoDataFrame(
            gdf_boundaries_regions,
            geometry=col_geometry
            )

        # ----- Additional setup -----
        # Assign colours:
        gdf_boundaries_regions = self.assign_colours_to_regions(
            gdf_boundaries_regions, 'region', col_colour)

        # ----- Save to self -----
        self.gdf_boundaries_regions = gdf_boundaries_regions
        # self.region_type = self.region_type

    def load_geometry_stroke_units(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.

        Import GeoDataFrame of selected stroke unit data.

        The file read is the output from Scenario() after the
        national stroke units have been reduced to those selected.

        The file contains coordinates (Easting, Northing) and (long, lat)
        which are picked out here and converted into Point() objects.
        The crs (coordinate reference system) is set to British National
        Grid.

        Inputs
        ------
        setup - Setup() object. Contains attributes for paths to the
                data directory and the geojson file names.

        Returns
        -------
        gdf_units - GeoDataFrame. One row per selected stroke unit in
                    the file. Columns include unit name and geometry.
        """
        # Selected stroke units names, coordinates, and services.
        try:
            df_units = self.df_units
        except AttributeError:
            self.load_run_data(['df_units'])
            df_units = self.df_units
        # Index column: Postcode.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        if self.data_type == 'combined':
            x_col = ('any', 'Easting')
            y_col = ('any', 'Northing')
            coords_col = ('any', 'geometry')
        else:
            x_col = 'Easting'
            y_col = 'Northing'
            coords_col = 'geometry'

        # Convert to geometry (point):
        # Create coordinates:
        # Current setup means sometimes these columns have different names.
        # TO DO - fix that please! ---------------------------------------------------
        # Extra .values.reshape are to remove the column headings.
        x = df_units[x_col].values.reshape(len(df_units))
        y = df_units[y_col].values.reshape(len(df_units))
        crs = 'EPSG:27700'  # by definition for easting/northing.

        df_units[coords_col] = geopandas.points_from_xy(x, y)

        # Convert to GeoDataFrame:
        gdf_units = geopandas.GeoDataFrame(
            df_units, geometry=coords_col, crs=crs
        )
        # # Convert to British National Grid coordinates if necessary:
        # if crs != 'EPSG:27700':
        #     gdf_units = gdf_units.to_crs('EPSG:27700')

        self.gdf_points_units = gdf_units

    def load_geometry_transfer_units(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # Selected stroke units names, coordinates, and services.
        try:
            df_transfer = self.df_transfer
        except AttributeError:
            self.load_run_data(['df_transfer'])
            df_transfer = self.df_transfer
        # Index column:
        #   - combined: ['Postcode', 'name_nearest_MT']
        #   - separate: 'Postcode'
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        if self.data_type == 'combined':
            x_col = ('any', 'Easting')
            y_col = ('any', 'Northing')
            x_col_mt = ('any', 'Easting_mt')
            y_col_mt = ('any', 'Northing_mt')
            col_unit = ('any', 'unit_coords')
            col_tran = ('any', 'transfer_coords')
            col_line_coords = ('any', 'line_coords')
            col_line_geometry = ('any', 'geometry')
        else:
            x_col = 'Easting'
            y_col = 'Northing'
            x_col_mt = 'Easting_mt'
            y_col_mt = 'Northing_mt'
            col_unit = 'unit_coords'
            col_tran = 'transfer_coords'
            col_line_coords = 'line_coords'
            col_line_geometry = 'geometry'

        # Convert to geometry (line):

        # Make a column of coordinates [x, y]:
        xy = df_transfer[[x_col, y_col]]
        df_transfer[col_unit] = xy.values.tolist()

        xy = df_transfer[[x_col_mt, y_col_mt]]
        df_transfer[col_tran] = xy.values.tolist()

        gdf_transfer = self.create_lines_from_coords(
            df_transfer,
            [col_unit, col_tran],
            col_line_coords,
            col_line_geometry
            )

        self.gdf_lines_transfer = gdf_transfer

    # ############################
    # ##### HELPER FUNCTIONS #####
    # ############################
    def create_lines_from_coords(
            self,
            df,
            cols_with_coords,
            col_coord,
            col_geom
            ):
        """
        Convert DataFrame with coords to GeoDataFrame with LineString.

        Initially group coordinates from multiple columns into one:
        +--------+--------+       +------------------+
        |  col1  |  col2  |       |   line_coords    |
        +--------+--------+  -->  +------------------+
        | [a, b] | [c, d] |       | [[a, b], [c, d]] |
        +--------+--------+       +------------------+
        And convert the single column into shapely.LineString objects
        with associated crs. Then convert the input DataFrame into
        a GeoDataFrame with the new Line objects.

        Inputs
        ------
        df               - pd.DataFrame. Contains some coordinates.
        cols_with_coords - list. List of column names in df that contain
                        coordinates for making lines.

        Returns
        -------
        gdf - GeoDataFrame. The input df with the new Line
            geometry objects and converted to a GeoDataFrame.
        """
        # Combine multiple columns of coordinates into a single column
        # with a list of lists of coordinates:
        df[col_coord] = df[cols_with_coords].values.tolist()

        # Drop any duplicates:
        df = df.drop_duplicates(col_coord)

        # Convert line coords to LineString objects:
        df[col_geom] = [
            LineString(coords) for coords in df[col_coord]]

        # Convert to GeoDataFrame:
        gdf = geopandas.GeoDataFrame(
            df, geometry=col_geom  #, crs="EPSG:4326"
        )
        # if isinstance(col_geom, tuple):
        #     gdf['geometry'] 
        # TO DO - implement CRS explicitly ---------------------------------------------
        return gdf

    def create_combo_cols(self, gdf, scenario):
        """
        TO DO - write me
        When dataframe doesn't have the diff_this_minus_that columns,
        use this function to create that data and prevent KeyError later.
        """
        # Find out what diff what:
        scen_bits = scenario.split('_')
        # Assume scenario = diff_scenario1_minus_scenario2:
        scen1 = scen_bits[1]
        scen2 = scen_bits[3]

        cols_to_combine = gdf[scen1].columns.to_list()
        for col in cols_to_combine:
            if isinstance(col, tuple):
                scen_col = (scenario, *col)
                scen1_col = (scen1, *col)
                scen2_col = (scen2, *col)
            else:
                # Assume it's a string.
                scen_col = (scenario, col)
                scen1_col = (scen1, col)
                scen2_col = (scen2, col)
            gdf[scen_col] = gdf[[scen1_col, scen2_col]].max(axis='columns')
        return gdf

    def assign_colours_to_regions(self, gdf, region_type, col_col):
        """
        wip, this version pretty useless.
        """

        colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

        # Use any old colours as debug:
        np.random.seed(42)
        colour_arr = np.random.choice(colours, size=len(gdf))

        # Add to the DataFrame:
        gdf[col_col] = colour_arr

        return gdf

    def assign_colours_to_regions_BROKEN(self, gdf, region_type):

        colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

        # TO DO - this neighbours function isn't working right -----------------------------
        # currently says that Gloucestershire doesn't border Bristol or Bath regions -------
        gdf = self.find_neighbours_for_regions(gdf, self.region_type)
        gdf = gdf.sort_values('total_neighbours', ascending=False)

        neighbour_list = gdf[self.region_type].tolist()

        neighbour_grid = np.full((len(gdf), len(gdf)), False)
        for row, neighbour_list_here in enumerate(gdf['neighbour_list']):
            for n in neighbour_list_here:
                col = neighbour_list.index(n)
                neighbour_grid[row, col] = True
                neighbour_grid[col, row] = True

        # Make a grid. One column per colour, one row per region.
        colour_grid = np.full((len(gdf), len(colours)), True)
        # To index row x: colour_grid[x, :]
        # To index col x: colour_grid[:, x]

        for row, region in enumerate(neighbour_list):
            # Which colours can this be?
            colours_here = colour_grid[row, :]

            # Pick the first available colour.
            ind_to_pick = np.where(colours_here == True)[0][0]

            # Update its neighbours' colour information.
            rows_neighbours = np.where(neighbour_grid[row, :] == True)[0]
            # Only keep these rows when we haven't checked them yet:
            rows_neighbours = [r for r in rows_neighbours if r > row]
            colour_grid[rows_neighbours, ind_to_pick] = False

            # Update its own colour information.
            colour_grid[row, :] = False
            colour_grid[row, ind_to_pick] = True

        # Use the bool colour grid to assign colours:
        colour_arr = np.full(len(neighbour_list), colours[0], dtype=object)
        for i, colour in enumerate(colours):
            colour_arr[np.where(colour_grid[:, i] == True)] = colour

        # Add to the DataFrame:
        gdf['colour'] = colour_arr

        # Use any old colours as debug:
        # np.random.seed(42)
        # colour_arr = np.random.choice(colours, size=len(gdf))

        return gdf

    def round_coordinates(self, df_geojson):
        from shapely.geometry import shape, mapping  # For conversion from shapely polygon to geojson and back
        # Remove floating point errors
        for i in range(len(df_geojson)):
            poly = df_geojson['geometry'][i]
            # Convert shapely object to geojson object
            gpoly = mapping(poly)
            if len(gpoly['coordinates']) == 1:
                # This is probably a normal polygon.
                a_coords = np.array(gpoly['coordinates'])
                new_coords = np.round(a_coords, 3)
            else:
                # This is probably a multipolygon but could be a polygon
                # that has multiple sets of coordinates for some reason
                # (maybe a hole, doughnut-shaped polygon?).
                new_coords = []
                for c, coords in enumerate(gpoly['coordinates']):
                    a_coords = np.array(gpoly['coordinates'][c])
                    a_coords = np.round(a_coords, 3)
                    new_coords.append(a_coords)
            gpoly['coordinates'] = new_coords

            # Convert back to shapely object
            poly = shape(gpoly)
            # Place back into the DataFrame
            df_geojson['geometry'][i] = poly
            return df_geojson

    def find_neighbours_for_regions(self, df_geojson, region_type='ICG22NM'):
        import shapely

        def split_multipolygons(df_geojson):
            # Expand the dataframe - need a separate row for each polygon, so split apart any multipolygons.
            df = df_geojson.copy()
            df_new = pd.DataFrame(columns=df.columns)

            r = 0
            for i in range(len(df)):
                row = df.iloc[i]
                if isinstance(row['geometry'], shapely.geometry.polygon.Polygon):
                    # All ok here, copy row contents exactly:
                    df_new.loc[r] = df.iloc[i]
                    r += 1
                else:
                    # MultiPolygon! Split it.
                    # Place each Polygon on its own row.
                    multipoly = row['geometry']
                    for poly in list(multipoly.geoms):
                        row_new = row.copy()
                        row_new['geometry'] = poly
                        df_new.loc[r] = row_new
                        r += 1

            # Convert this to a GeoDataFrame to match the input df_geojson:
            df_new = geopandas.GeoDataFrame(
                df_new, geometry=df_new['geometry']#, crs="EPSG:4326"
            )
            return df_new

        def find_neighbours(df_new, region_type):
            df = df_new.copy()
            df['my_neighbors'] = [[]] * len(df)

            for index, row in df.iterrows():
                if isinstance(row['geometry'], shapely.geometry.polygon.Polygon): 
                    neighbors = df[df.geometry.touches(row['geometry'])][region_type].tolist()
                elif not isinstance(row['geometry'], shapely.geometry.point.Point):
                    # This is a MultiPolygon. Check each Polygon separately.
                    multipoly = row['geometry']
                    neighbors = []
                    for polygon in list(multipoly.geoms):
                        neighbours_here = df[df.geometry.intersects(polygon)].index.tolist()
                        neighbors += neighbours_here
                else:
                    # It's a point! Ignore.
                    pass
                try:
                    # Don't let the place be its own neighbour.
                    neighbors = neighbors.remove(row.name)
                except ValueError:
                    # Its own name is not in the list of neighbours.
                    pass
                df.at[index, 'my_neighbors'] = neighbors
                # df.loc[index]['my_neighbors'] = neighbors #", ".join([f'{n}' for n in neighbors])

            df_neighbours = df.copy()
            return df_neighbours

        def unsplit_multipolygons(df_geojson, df_neighbours):
            df = df_geojson.copy()
            df['neighbour_list'] = [[]] * len(df)

            for i in range(len(df)):
                # What is this called in the original dataframe?
                objectid = df.iloc[i]['OBJECTID']
                # region_name = df.iloc[i]['CCG19CD']
                # Where did this end up in the polygon dataframe?
                df_here = df_neighbours[df_neighbours['OBJECTID'] == objectid]
                # Combine multiple lists of neighbours into one list:
                list_of_neighbours = np.concatenate(df_here['my_neighbors'].values)
                # Remove any repeats:
                list_of_neighbours = list(set(list_of_neighbours))
                df.at[i, 'neighbour_list'] = list_of_neighbours
            return df

        df_geojson = self.round_coordinates(df_geojson)
        df_new = split_multipolygons(df_geojson)
        df_neighbours = find_neighbours(df_new, self.region_type)
        df = unsplit_multipolygons(df_geojson, df_neighbours)

        # Record the number of neighbours:
        df['total_neighbours'] = df['neighbour_list'].str.len()
        return df

    def get_selected_area_extent(
            self,
            gdf_selected,
            leeway=20000,
            ):
        """
        # What is the extent of the selected regions?
        """
        minx, miny, maxx, maxy = gdf_selected.geometry.total_bounds
        # Give this some leeway:
        minx -= leeway
        miny -= leeway
        maxx += leeway
        maxy += leeway
        map_extent = [minx, maxx, miny, maxy]
        # Turn the points into a box:
        box = Polygon((
            (minx, miny),
            (minx, maxy),
            (maxx, maxy),
            (maxx, miny),
            (minx, miny),
        ))
        return box, map_extent

    # #########################
    # ##### PLOT WRAPPERS #####
    # #########################
    def plot_map_selected_regions(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected units.
        """
        map_args, map_kwargs = self._setup_plot_map_selected_regions(
            scenario,
            save
            )
        self._plt_plot_map_selected_regions(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_selected_units(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected units.
        """

        map_args, map_kwargs = self._setup_plot_map_selected_units(
            scenario,
            save
            )
        self._plt_plot_map_selected_units(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_catchment(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected unit catchments.
        """
        map_args, map_kwargs = self._setup_plot_map_catchment(
            scenario,
            save=save
            )
        self._plt_plot_map_catchment(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_outcome(
            self,
            scenario: str,
            outcome: str,
            save=True,
            region=None,
            boundary_kwargs={},
            ):
        """
        Wrangle data and plot a map of LSOA outcomes.
        """
        if region is None:
            pass
        else:
            # Reload the region data.
            self.region_type = region
            self.load_geometry_regions(region)

        map_args, map_kwargs = self._setup_plot_map_outcome(
            scenario,
            outcome,
            boundary_kwargs=boundary_kwargs,
            save=save
            )
        self._plt_plot_map_outcome(
            *map_args,
            **map_kwargs,
            title=f'{scenario}\n{outcome}',
            save=save
        )

    # ###########################
    # ##### SETUP FOR PLOTS #####
    # ###########################
    def _setup_plot_map_selected_regions(
            self,
            scenario: str,
            save=True
            ):
        # Load in reference data for this scenario
        # (always ignore "combined"):
        self.load_run_data(['df_regions', 'df_units'], dir_data=scenario)

        # If the geometry data doesn't exist, load it in:
        data_dict = {
            'gdf_boundaries_regions': self.load_geometry_regions,
            'gdf_points_units': self.load_geometry_stroke_units
        }
        for attr, func in data_dict.items():
            try:
                self.getattr(attr)
            except AttributeError:
                func()

        if self.data_type == 'combined':
            # This shouldn't run... for now.
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c].copy()
                gdf_points_units = self.gdf_points_units[c].copy()
            except KeyError:
                # The scenario isn't in the Data.
                # TO DO - proper error message here ---------------------------------------------
                print('oh no')
                raise KeyError('Scenario is missing.') from None
            # Remove the excess column heading:
            # TO DO - set up column level name 'scenario' here -----------------------------------
            gdf_boundaries_regions = (
                gdf_boundaries_regions.droplevel(0, axis='columns'))
            gdf_points_units = (
                gdf_points_units.droplevel(0, axis='columns'))
            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            g = 'geometry'
            gdf_boundaries_regions = gdf_boundaries_regions.set_geometry(g)
            gdf_points_units = gdf_points_units.set_geometry(g)
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()

        box, map_extent = self.get_selected_area_extent(
            gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1],
            leeway=20000,
            )

        # Which other regions are contained in this bounding box?
        mask = gdf_boundaries_regions.geometry.intersects(box)
        gdf_boundaries_regions = gdf_boundaries_regions[mask]
        # Which stroke units are contained in this bounding box?
        mask = gdf_points_units.geometry.intersects(box)
        gdf_points_units = gdf_points_units[mask]

        # Restrict polygon geometry to the edges of the box.
        gdf_boundaries_regions['geometry'] = (
            gdf_boundaries_regions.geometry.intersection(box))

        # Add a label number for each boundary.
        gdf_boundaries_regions = gdf_boundaries_regions.sort_values(
            ['selected', 'BNG_N'], ascending=False
        )
        gdf_boundaries_regions['label'] = np.arange(
            1, len(gdf_boundaries_regions) + 1).astype(str)
        # Get coordinates for each label:
        # point_label = []
        # for poly in gdf_boundaries_regions.geometry:
        #     point = poly.representative_point()
        #     point_label.append(point)
        point_label = ([poly.representative_point() for
                        poly in gdf_boundaries_regions.geometry])
        gdf_boundaries_regions['point_label'] = point_label

        # Which stroke units are in the selected regions?
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        mask = gdf_points_units['region'].isin(regions_selected)
        gdf_points_units['region_selected'] = 0
        gdf_points_units.loc[mask, 'region_selected'] = 1
        # Add a label letter for each unit.
        gdf_points_units = gdf_points_units.sort_values(
            ['region_selected', 'Northing'], ascending=False
        )
        import string
        # List ['A', 'B', 'C', ..., 'Z']:
        str_labels = list(string.ascii_uppercase)
        if len(str_labels) < len(gdf_points_units):
            # Add more letters at the end starting with
            # ['AA', 'AB', 'AC', ... 'AZ'].
            i = 0
            str_labels_orig = list(string.ascii_uppercase)
            while len(str_labels) < len(gdf_points_units):
                str_labels2 = [f'{str_labels[i]}{s}' for s in str_labels_orig]
                str_labels += str_labels2
                i += 1
        else:
            pass
        gdf_points_units['label'] = str_labels[:len(gdf_points_units)]

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'colour',                               # background colour
            'selected',                             # line type selection
            'label',                                # label annotation
            'point_label',                          # label position
            'region',                               # legend label
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'Use_IVT', 'Use_MT', 'Use_MSU',         # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'region_selected'                       # label kwargs
            ]]

        # Create file name:
        if save:
            dir_output = ''
            if self.data_type == 'combined':
                dir_output = self.setup.dir_output_combined
            else:
                try:
                    for d in self.setup.list_dir_output:
                        end = os.path.split(d)[-1]
                        if end == scenario:
                            dir_output = d
                except AttributeError:
                    # Setup is not defined.
                    pass

            file_name = f'map_selected_regions_{scenario}.jpg'
            path_to_file = os.path.join(dir_output, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_regions,
            gdf_points_units
            )
        map_kwargs = dict(
            map_extent=map_extent,
            path_to_file=path_to_file
            )
        return map_args, map_kwargs

    def _setup_plot_map_selected_units(
            self,
            scenario,
            save=True
            ):

        # Load in reference data for this scenario
        # (always ignore "combined"):
        self.load_run_data(
            ['df_regions', 'df_units', 'df_transfer'],
            dir_data=scenario
            )

        # If the geometry data doesn't exist, load it in:
        data_dict = {
            'gdf_boundaries_regions': self.load_geometry_regions,
            'gdf_points_units': self.load_geometry_stroke_units,
            'gdf_lines_transfer': self.load_geometry_transfer_units,
        }
        for attr, func in data_dict.items():
            try:
                self.getattr(attr)
            except AttributeError:
                func()

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c].copy()
                gdf_points_units = self.gdf_points_units[c].copy()
                gdf_lines_transfer = self.gdf_lines_transfer[c].copy()
            except KeyError:
                # The scenario isn't in the Data.
                # TO DO - proper error message here ---------------------------------------------
                print('oh no')
                raise KeyError('Scenario is missing.') from None
            # Remove the excess column heading:
            # TO DO - set up column level name 'scenario' here -----------------------------------
            gdf_boundaries_regions = (
                gdf_boundaries_regions.droplevel(0, axis='columns'))
            gdf_points_units = (
                gdf_points_units.droplevel(0, axis='columns'))
            gdf_lines_transfer = (
                gdf_lines_transfer.droplevel(0, axis='columns'))
            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            g = 'geometry'
            gdf_boundaries_regions = gdf_boundaries_regions.set_geometry(g)
            gdf_points_units = gdf_points_units.set_geometry(g)
            gdf_lines_transfer = gdf_lines_transfer.set_geometry(g)
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()
            gdf_lines_transfer = self.gdf_lines_transfer.copy()

        box, map_extent = self.get_selected_area_extent(
            gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1],
            leeway=20000,
            )

        # TO DO - function this --------------------------------------------------------?
        # Which other regions are contained in this bounding box?
        mask = gdf_boundaries_regions.geometry.intersects(box)
        gdf_boundaries_regions = gdf_boundaries_regions[mask]
        # Which stroke units are contained in this bounding box?
        mask = gdf_points_units.geometry.intersects(box)
        gdf_points_units = gdf_points_units[mask]

        # TO DO - function this --------------------------------------------------------?
        # Restrict polygon geometry to the edges of the box.
        gdf_boundaries_regions['geometry'] = (
            gdf_boundaries_regions.geometry.intersection(box))

        # TO DO - function this --------------------------------------------------------
        # Which stroke units are in the selected regions?
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        mask = gdf_points_units['region'].isin(regions_selected)
        gdf_points_units['region_selected'] = 0
        gdf_points_units.loc[mask, 'region_selected'] = 1
        # Add a label letter for each unit.
        gdf_points_units = gdf_points_units.sort_values(
            ['region_selected', 'Northing'], ascending=False
        )
        import string
        # List ['A', 'B', 'C', ..., 'Z']:
        str_labels = list(string.ascii_uppercase)
        if len(str_labels) < len(gdf_points_units):
            # Add more letters at the end starting with
            # ['AA', 'AB', 'AC', ... 'AZ'].
            i = 0
            str_labels_orig = list(string.ascii_uppercase)
            while len(str_labels) < len(gdf_points_units):
                str_labels2 = [f'{str_labels[i]}{s}' for s in str_labels_orig]
                str_labels += str_labels2
                i += 1
        else:
            pass
        gdf_points_units['label'] = str_labels[:len(gdf_points_units)]

        gdf_lines_transfer = gdf_lines_transfer.reset_index().copy()
        # Set 'Use' to 1 when either the start or end unit
        # is in 'region_selected':
        gdf_lines_transfer['Use'] = 0
        df_units_rs = gdf_points_units.copy()
        df_units_rs = df_units_rs.reset_index()
        # Is start unit in region_selected?
        gdf_lines_transfer = pd.merge(
            gdf_lines_transfer,
            df_units_rs[['Postcode', 'region_selected']],
            left_on='Postcode', right_on='Postcode', how='left'
        )
        # Is end unit in region_selected?
        gdf_lines_transfer = pd.merge(
            gdf_lines_transfer,
            df_units_rs[['Postcode', 'region_selected']],
            left_on='name_nearest_MT', right_on='Postcode', how='left',
            suffixes=(None, '_MT')
        )
        gdf_lines_transfer['Use'][(
            (gdf_lines_transfer['region_selected'] == 1) |
            (gdf_lines_transfer['region_selected_MT'] == 1)
            )] = 1

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'colour',                               # background colour
            'selected',                             # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'Use_IVT', 'Use_MT', 'Use_MSU',         # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'region_selected'                       # label kwargs
            ]]
        gdf_lines_transfer = gdf_lines_transfer[[
            'geometry',                             # line end points
            'Use'                                   # line selection
            ]]

        # Create file name:
        if save:
            dir_output = ''
            if self.data_type == 'combined':
                dir_output = self.setup.dir_output_combined
            else:
                try:
                    for d in self.setup.list_dir_output:
                        end = os.path.split(d)[-1]
                        if end == scenario:
                            dir_output = d
                except AttributeError:
                    # Setup is not defined.
                    pass

            file_name = f'map_selected_units_{scenario}.jpg'
            path_to_file = os.path.join(dir_output, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer
        )
        map_kwargs = dict(
            path_to_file=path_to_file,
            map_extent=map_extent
            )
        return map_args, map_kwargs

    def _setup_plot_map_catchment(
            self,
            scenario: str,
            boundary_kwargs={},
            save=True
            ):
        """

        """
        # Load in reference data for this scenario
        # (always ignore "combined"):
        self.load_run_data(
            ['df_lsoa', 'df_regions', 'df_units', 'df_transfer'],
            dir_data=scenario
            )

        # If the geometry data doesn't exist, load it in:
        data_dict = {
            'gdf_boundaries_regions': self.load_geometry_regions,
            'gdf_boundaries_lsoa': self.load_geometry_lsoa,
            'gdf_points_units': self.load_geometry_stroke_units,
            'gdf_lines_transfer': self.load_geometry_transfer_units,
        }
        for attr, func in data_dict.items():
            try:
                self.getattr(attr)
            except AttributeError:
                func()

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_lsoa = self.gdf_boundaries_lsoa[c]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c]
                gdf_points_units = self.gdf_points_units[c]
                gdf_lines_transfer = self.gdf_lines_transfer[c]
            except KeyError:
                # The scenario isn't in the Data.
                # TO DO - proper error message here
                print('oh no')
                raise KeyError('Scenario is missing.') from None

            # Remove the excess column headings:
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                0, axis='columns')
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                1, axis='columns')
            gdf_boundaries_regions = gdf_boundaries_regions.droplevel(
                0, axis='columns')
            gdf_points_units = gdf_points_units.droplevel(
                0, axis='columns')
            gdf_lines_transfer = gdf_lines_transfer.droplevel(
                0, axis='columns')

            # TO DO - FUCNTION THIS PLEASE
            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            g = 'geometry'
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_geometry(g)
            gdf_boundaries_regions = gdf_boundaries_regions.set_geometry(g)
            gdf_points_units = gdf_points_units.set_geometry(g)
            gdf_lines_transfer = gdf_lines_transfer.set_geometry(g)

        else:
            gdf_boundaries_lsoa = self.gdf_boundaries_lsoa
            gdf_boundaries_regions = self.gdf_boundaries_regions
            gdf_points_units = self.gdf_points_units
            gdf_lines_transfer = self.gdf_lines_transfer

            # Remove the excess column headings:
            # TO DO - change this to drop by level name -------------------------
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                1, axis='columns')
            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            g = 'geometry'
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_geometry(g)


        # Combine LSOA geometry - from separate polygon per LSOA to one
        # big polygon for all LSOAs in catchment area.
        gdf_boundaries_lsoa = gdf_boundaries_lsoa.reset_index()
        #  TO DO - might not always be called _IVT in following column:
        gdf_boundaries_lsoa_glob = gdf_boundaries_lsoa[['postcode_nearest_IVT', 'geometry']].dissolve(by='postcode_nearest_IVT')
        # Overwrite existing name:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa_glob
        # ^ check if above name fudge is still necessary - TO DO -----------------------------

        gdf_combo = pd.concat(
            (gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1].reset_index()['geometry'],
             gdf_boundaries_lsoa.reset_index()['geometry']),
            axis='rows'
        )

        # TO DO - currently this means that stroke units get different labels between 
        # this map and the other stroke unit selection map. Fix it! ------------------------

        # Take map extent from the combined LSOA and region geometry.
        box, map_extent = self.get_selected_area_extent(
            gdf_combo,
            leeway=20000,
            )

        # TO DO - function this --------------------------------------------------------?
        # Which other regions are contained in this bounding box?
        mask = gdf_boundaries_regions.geometry.intersects(box)
        gdf_boundaries_regions = gdf_boundaries_regions[mask]
        # Which stroke units are contained in this bounding box?
        mask = gdf_points_units.geometry.intersects(box)
        gdf_points_units = gdf_points_units[mask]

        # TO DO - function this --------------------------------------------------------?
        # Restrict polygon geometry to the edges of the box.
        gdf_boundaries_regions['geometry'] = (
            gdf_boundaries_regions.geometry.intersection(box))

        # TO DO - function this --------------------------------------------------------
        # Which stroke units are in the selected regions?
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        mask = gdf_points_units['region'].isin(regions_selected)
        gdf_points_units['region_selected'] = 0
        gdf_points_units.loc[mask, 'region_selected'] = 1
        # Add a label letter for each unit.
        gdf_points_units = gdf_points_units.sort_values(
            ['region_selected', 'Northing'], ascending=False
        )
        import string
        # List ['A', 'B', 'C', ..., 'Z']:
        str_labels = list(string.ascii_uppercase)
        if len(str_labels) < len(gdf_points_units):
            # Add more letters at the end starting with
            # ['AA', 'AB', 'AC', ... 'AZ'].
            i = 0
            str_labels_orig = list(string.ascii_uppercase)
            while len(str_labels) < len(gdf_points_units):
                str_labels2 = [f'{str_labels[i]}{s}' for s in str_labels_orig]
                str_labels += str_labels2
                i += 1
        else:
            pass
        gdf_points_units['label'] = str_labels[:len(gdf_points_units)]

        gdf_lines_transfer = gdf_lines_transfer.reset_index().copy()
        # Set 'Use' to 1 when either the start or end unit
        # is in 'region_selected':
        gdf_lines_transfer['Use'] = 0
        df_units_rs = gdf_points_units.copy()
        df_units_rs = df_units_rs.reset_index()
        # Is start unit in region_selected?
        gdf_lines_transfer = pd.merge(
            gdf_lines_transfer,
            df_units_rs[['Postcode', 'region_selected']],
            left_on='Postcode', right_on='Postcode', how='left'
        )
        # Is end unit in region_selected?
        gdf_lines_transfer = pd.merge(
            gdf_lines_transfer,
            df_units_rs[['Postcode', 'region_selected']],
            left_on='name_nearest_MT', right_on='Postcode', how='left',
            suffixes=(None, '_MT')
        )
        gdf_lines_transfer['Use'][(
            (gdf_lines_transfer['region_selected'] == 1) |
            (gdf_lines_transfer['region_selected_MT'] == 1)
            )] = 1

        # TO DO - make gdf_boundaries_regions contains_selected_lsoa column.

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[[
            'geometry',                             # shapes
            ]]
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            # 'contains_selected_lsoa',               # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'Use_IVT', 'Use_MT', 'Use_MSU',# 'Use',  # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'region_selected'                       # label kwargs
            ]]
        gdf_lines_transfer = gdf_lines_transfer[[
            'geometry',                             # line end points
            'Use'                                   # line selection
        ]]

        lsoa_boundary_kwargs = {
            # 'column': 'postcode_nearest_IVT', # TEMP - TO DO - might change column name
            'cmap': 'Blues',
            'edgecolor': 'face'
        }
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        if save:
            dir_output = ''
            if self.data_type == 'combined':
                dir_output = self.setup.dir_output_combined
            else:
                try:
                    for d in self.setup.list_dir_output:
                        end = os.path.split(d)[-1]
                        if end == scenario:
                            dir_output = d
                except AttributeError:
                    # Setup is not defined.
                    pass

            file_name = f'map_catchment_{scenario}.jpg'

            path_to_file = os.path.join(dir_output, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer
            )
        map_kwargs = dict(
            lsoa_boundary_kwargs=lsoa_boundary_kwargs,
            map_extent=map_extent,
            path_to_file=path_to_file
        )
        return map_args, map_kwargs

    def _setup_plot_map_outcome(
            self,
            scenario: str,
            outcome: str,
            boundary_kwargs={},
            save=True
            ):
        """

        """
        # If the geometry data doesn't exist, load it in:
        data_dict = {
            'gdf_boundaries_regions': self.load_geometry_regions,
            'gdf_boundaries_lsoa': self.load_geometry_lsoa,
            'gdf_points_units': self.load_geometry_stroke_units,
        }
        for attr, func in data_dict.items():
            try:
                self.getattr(attr)
            except AttributeError:
                func()

        if self.data_type == 'combined':
            try:
                gdf_boundaries_lsoa = self.gdf_boundaries_lsoa
            except KeyError:
                # TO DO - proper error message here
                print('oh no')
                raise KeyError('LSOA are missing?') from None

            # Find shared outcome limits.
            # Take only scenarios containing 'diff':
            mask = (
                (gdf_boundaries_lsoa.columns.get_level_values(0).str.contains('diff'))
            )
            if outcome.startswith('diff'):
                pass
            else:
                # Take the opposite condition, take only scenarios
                # not containing 'diff'.
                mask = ~mask

            mask = (
                mask &
                (gdf_boundaries_lsoa.columns.get_level_values(2) == 'mean') &
                (gdf_boundaries_lsoa.columns.get_level_values(1) == outcome)
            )
            all_mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
            vlim_abs = all_mean_vals.abs().max().values[0]
            vmax = all_mean_vals.max().values[0]
            vmin = all_mean_vals.min().values[0]

            try:
                gdf_boundaries_regions = self.gdf_boundaries_regions
                gdf_points_units = self.gdf_points_units

                if 'diff' in scenario:
                    # Add any other columns that these expect.
                    gdf_boundaries_regions = self.create_combo_cols(
                        gdf_boundaries_regions, scenario)
                    gdf_points_units = self.create_combo_cols(
                        gdf_points_units, scenario)

                # # Remove excess scenario data:
                s = ['any', scenario]
                gdf_boundaries_lsoa = gdf_boundaries_lsoa[s]
                gdf_boundaries_regions = gdf_boundaries_regions[s]
                gdf_points_units = gdf_points_units[s]

            except KeyError:
                # The scenario isn't in the Data.
                # TO DO - proper error message here
                print('oh no')
                raise KeyError('Scenario is missing.') from None

            # Remove the "std" columns.
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
                gdf_boundaries_lsoa.iloc[:, (gdf_boundaries_lsoa.columns.get_level_values(2) == 'std')].columns,
                axis='columns'
            )

            # Remove the excess column headings:
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                0, axis='columns')
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                1, axis='columns')
            gdf_boundaries_regions = gdf_boundaries_regions.droplevel(
                0, axis='columns')
            gdf_points_units = gdf_points_units.droplevel(
                0, axis='columns')

            # TO DO - FUCNTION THIS PLEASE
            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_geometry('geometry')
            gdf_boundaries_regions = gdf_boundaries_regions.set_geometry('geometry')
            gdf_points_units = gdf_points_units.set_geometry('geometry')

        else:
            gdf_boundaries_lsoa = self.gdf_boundaries_lsoa
            gdf_boundaries_regions = self.gdf_boundaries_regions
            gdf_points_units = self.gdf_points_units

            # Find the colour limits.
            mask = (
                (gdf_boundaries_lsoa.columns.get_level_values(1) == 'mean') &
                (gdf_boundaries_lsoa.columns.get_level_values(0) == outcome)
            )
            mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
            vlim_abs = mean_vals.abs().max().values[0]
            vmax = mean_vals.max().values[0]
            vmin = mean_vals.min().values[0]

            # Remove the "std" columns.
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
                gdf_boundaries_lsoa.iloc[:, (gdf_boundaries_lsoa.columns.get_level_values(1) == 'std')].columns,
                axis='columns'
            )

            # Remove the excess column headings:
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                1, axis='columns')

            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_geometry('geometry')

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[[
            'geometry',  # shapes
            outcome      # colours
            ]]
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                # shapes
            'contains_selected_unit',  # line type selection
            'contains_selected_lsoa',  # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'Use_IVT', 'Use_MT', 'Use_MSU', #'Use',  # point selection
            'Hospital_name'                         # labels
            ]]

        lsoa_boundary_kwargs = {
            'column': outcome,
            'edgecolor': 'face',
            # Adjust size of colourmap key, and add label
            'legend_kwds': {
                'shrink': 0.5,
                'label': outcome
                },
            # Set to display legend
            'legend': True,
            }

        cbar_diff = True if outcome.startswith('diff') else False
        if cbar_diff:
            lsoa_boundary_kwargs['cmap'] = 'seismic'
            lsoa_boundary_kwargs['vmin'] = -vlim_abs
            lsoa_boundary_kwargs['vmax'] = vlim_abs
        else:
            cmap = 'plasma'
            if outcome == 'mRS shift':
                # Reverse the colourmap because lower values
                # are better for this outcome.
                cmap += '_r'
            lsoa_boundary_kwargs['cmap'] = cmap
            lsoa_boundary_kwargs['vmin'] = vmin
            lsoa_boundary_kwargs['vmax'] = vmax
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        if save:
            dir_output = ''
            if self.data_type == 'combined':
                dir_output = self.setup.dir_output_combined
            else:
                try:
                    for d in self.setup.list_dir_output:
                        end = os.path.split(d)[-1]
                        if end == scenario:
                            dir_output = d
                except AttributeError:
                    # Setup is not defined.
                    pass

            file_name = f'map_{outcome}_{scenario}_{self.region_type}.jpg'
            path_to_file = os.path.join(dir_output, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units
        )
        map_kwargs = dict(
            lsoa_boundary_kwargs=lsoa_boundary_kwargs,
            path_to_file=path_to_file
        )
        return map_args, map_kwargs

    # #######################
    # ##### PYPLOT MAPS #####
    # #######################
    def _plt_plot_map_selected_regions(
            self,
            gdf_boundaries_regions,
            gdf_points_units,
            map_extent=[],
            path_to_file='',
            save=True,
            show=False
            ):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax, extra_artists = maps.plot_map_selected_regions(
            gdf_boundaries_regions,
            gdf_points_units,
            ax=ax,
            map_extent=map_extent
        )

        if save:
            # Return extra artists so that bbox_inches='tight' line
            # in savefig() doesn't cut off the legends.
            # Adding legends with ax.add_artist() means that the
            # bbox_inches='tight' line ignores them.
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight'
                )
            plt.close()
        elif show:
            plt.show()
        else:
            # Don't do anything.
            plt.close()

    def _plt_plot_map_selected_units(
            self,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            map_extent=[],
            path_to_file='',
            save=True,
            show=False
            ):
        fig, ax = plt.subplots(figsize=(10, 10))

        ax, extra_artists = maps.plot_map_selected_units(
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            map_extent=map_extent,
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight'
                )
            plt.close()
        elif show:
            plt.show()
        else:
            # Don't do anything.
            plt.close()

    def _plt_plot_map_catchment(
            self,
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            title='',
            lsoa_boundary_kwargs={},
            map_extent=[],
            save=True,
            show=False,
            path_to_file=''
            ):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title)

        ax, extra_artists = maps.plot_map_catchment(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            map_extent=map_extent,
            boundary_kwargs=lsoa_boundary_kwargs
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plt_plot_map_outcome(
            self,
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            title='',
            lsoa_boundary_kwargs={},
            save=True,
            path_to_file=None
            ):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title)

        ax = maps.plot_map_outcome(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            ax=ax,
            boundary_kwargs=lsoa_boundary_kwargs
        )

        if save:
            plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
