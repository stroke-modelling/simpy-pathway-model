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

"""
# TO DO
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.

from classes.setup import Setup

import classes.map_functions as maps  # for plotting.


class Map(object):
    """
    Combine files from multiple runs of the pathway.

    class Combine():

    TO DO - write me
    """
    def __init__(self, *initial_data, **kwargs):

        self.region_type = 'ICB22NM'

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
    def load_run_data(self, dir_data=None):
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
                'index_col': 0,
                },
            'df_regions': {
                'file': self.setup.file_combined_selected_regions,
                'header': [0, 1],
                'index_col': 0,
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
                'index_col': [0],
                },
            'df_lsoa': {
                'file': self.setup.file_selected_lsoas,
                'header': [0],
                'index_col': 0,
                },
            'df_regions': {
                'file': self.setup.file_selected_regions,
                'header': [0],
                'index_col': 0,
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
            dir_data = self.setup.dir_output_all_runs
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

        for label, data_dict in dicts_data.items():
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
                print(f'Cannot import {label} from {data_dict["file"]}')

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
            'CCG19NM': self.setup.file_geojson_ccg,
            'ICB22NM': self.setup.file_geojson_icb,
            'LAD17NM': self.setup.file_geojson_lad,
            'STP19NM': self.setup.file_geojson_stp,
            'LHB20NM': self.setup.file_geojson_lhb,
            'SCN17NM': self.setup.file_geojson_scn,
            'RGN11NM': self.setup.file_geojson_rgn,
        }
        # n.b. current setup as of January 2024 is that the dict
        # keys match the column names in the LSOA_regions.csv
        # and similar reference files. The actual geojson files
        # definitely contain the same type of region, but could
        # be from a different year than the one listed here.

        # Import region file:
        dir_input = self.setup.dir_data_geojson
        file_input = geojson_file_dict[region_type]
        path_to_file = os.path.join(dir_input, file_input)
        gdf_boundaries = geopandas.read_file(path_to_file)
        gdf_boundaries = gdf_boundaries.set_index(region_type)
        # If crs is given in the file, geopandas automatically
        # pulls it through. Convert to National Grid coordinates:
        if gdf_boundaries.crs != 'EPSG:27700':
            gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
        return gdf_boundaries

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
        df_lsoa = self.df_lsoa
        # Index column: LSOA11NM.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        # All LSOA shapes:
        gdf_boundaries_lsoa = self.import_geojson('LSOA11NM')
        # Index column: LSOA11NM.
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
            pass

        if self.data_type == 'combined':
            # Drop columns that are duplicated across DataFrames.
            df_lsoa = df_lsoa.drop([('any', 'LSOA11CD')], axis='columns')

            # Make all data to be combined have the same column levels.
            # TO DO - is there a more pandas way to do this?

            # LSOA names:
            df_lsoa_column_arr = np.array(
                [[n for n in c] for c in df_lsoa.columns])
            df_lsoa = pd.DataFrame(
                df_lsoa.values,
                index=df_lsoa.index,
                columns=[
                    df_lsoa_column_arr[:, 0],       # scenario
                    df_lsoa_column_arr[:, 1],       # property
                    [''] * len(df_lsoa.columns),    # subtype
                ]
            )
            # Geometry:
            gdf_boundaries_lsoa = pd.DataFrame(
                gdf_boundaries_lsoa.values,
                index=gdf_boundaries_lsoa.index,
                columns=[
                    ['any'] * len(gdf_boundaries_lsoa.columns),    # scenario
                    gdf_boundaries_lsoa.columns,                   # property
                    [''] * len(gdf_boundaries_lsoa.columns),       # subtype
                ]
            )
            # Results:
            # This already has the three column levels.

            # Then merge together all of the DataFrames.
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
                gdf_boundaries_lsoa.columns.set_names(
                    ['scenario', 'property', 'subtype']))

            # Sort the results by scenario (top column index):
            gdf_boundaries_lsoa = gdf_boundaries_lsoa[sorted(list(set(
                gdf_boundaries_lsoa.columns.get_level_values('scenario'))))]

            # Convert to GeoDataFrame:
            gdf_boundaries_lsoa = geopandas.GeoDataFrame(
                gdf_boundaries_lsoa,
                geometry=('any', 'geometry', '')
                )
            # gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
            #     ('any', 'geometry', ''), axis='columns')
        else:
            # Similar process but without the scenario row.
            # Drop columns that are duplicated across DataFrames.
            df_lsoa = df_lsoa.drop('LSOA11CD', axis='columns')

            # Make all data to be combined have the same column levels.
            # TO DO - is there a more pandas way to do this?

            # LSOA names:
            df_lsoa = pd.DataFrame(
                df_lsoa.values,
                index=df_lsoa.index,
                columns=[
                    df_lsoa.columns,                # property
                    [''] * len(df_lsoa.columns),    # subtype
                ]
            )
            # Geometry:
            gdf_boundaries_lsoa = pd.DataFrame(
                gdf_boundaries_lsoa.values,
                index=gdf_boundaries_lsoa.index,
                columns=[
                    gdf_boundaries_lsoa.columns,                # property
                    [''] * len(gdf_boundaries_lsoa.columns),    # subtype
                ]
            )
            # Results:
            # This already has the two column levels.

            # Then merge together all of the DataFrames.
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
                gdf_boundaries_lsoa.columns.set_names(
                    ['property', 'subtype']))

            # Convert to GeoDataFrame:
            gdf_boundaries_lsoa = geopandas.GeoDataFrame(
                gdf_boundaries_lsoa,
                geometry=('geometry', '')
                )
            # gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
            #     ('geometry', ''), axis='columns')

        self.gdf_boundaries_lsoa = gdf_boundaries_lsoa

    def load_geometry_regions(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # ----- Gather data -----
        # Selected regions names and usage.
        # ['{region type}', 'contains_selected_unit',
        #  'contains_selected_lsoa']
        df_regions = self.df_regions
        # Index column: {region type}.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        # All region polygons:
        gdf_boundaries_regions = self.import_geojson(self.region_type)
        # Index column: LSOA11NM.
        # Always has only one unnamed column index level.

        if self.data_type == 'combined':
            # Make all data to be combined have the same column levels.
            # TO DO - is there a more pandas way to do this?
            # Would save bungling the geodataframe/ crs lines.

            # Region names and usage:
            # This already has the two column levels.

            # Geometry:
            gdf_boundaries_regions = pd.DataFrame(
                gdf_boundaries_regions.values,
                index=gdf_boundaries_regions.index,
                columns=[
                    ['any'] * len(gdf_boundaries_regions.columns),    # scenario
                    gdf_boundaries_regions.columns,                   # property
                ]
            )

            # Then merge together all of the DataFrames.
            gdf_boundaries_regions = pd.merge(
                gdf_boundaries_regions, df_regions,
                left_index=True, right_index=True, how='right'
            )

            # Name the column levels:
            gdf_boundaries_regions.columns = (
                gdf_boundaries_regions.columns.set_names(
                    ['scenario', 'property']))

            # Sort the results by scenario (top column index):
            gdf_boundaries_regions = gdf_boundaries_regions[sorted(list(set(
                gdf_boundaries_regions.columns.get_level_values('scenario'))))]

            # Convert to GeoDataFrame:
            gdf_boundaries_regions = geopandas.GeoDataFrame(
                gdf_boundaries_regions,
                geometry=('any', 'geometry')
                )
            # gdf_boundaries_regions = gdf_boundaries_regions.drop(
            #     ('any', 'geometry'), axis='columns')

            # Assign colours:
            gdf_boundaries_regions = self.assign_colours_to_regions(
                gdf_boundaries_regions, self.region_type, ('any', 'colour'))
        else:
            # Similar process but without the scenario row.

            # Merge together all of the DataFrames.
            gdf_boundaries_regions = pd.merge(
                gdf_boundaries_regions, df_regions,
                left_index=True, right_index=True, how='right'
            )

            # Name the column levels:
            gdf_boundaries_regions.columns = (
                gdf_boundaries_regions.columns.set_names(['property']))

            # Convert to GeoDataFrame:
            gdf_boundaries_regions = geopandas.GeoDataFrame(
                gdf_boundaries_regions,
                geometry='geometry'
                )
            # gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
            #     ('geometry', ''), axis='columns')

            # Assign colours:
            gdf_boundaries_regions = self.assign_colours_to_regions(
                gdf_boundaries_regions, self.region_type, 'colour')

        self.gdf_boundaries_regions = gdf_boundaries_regions

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
    def create_lines_from_coords(self, df, cols_with_coords, col_coord, col_geom):
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

    def assign_colours_to_regions(self, gdf, region_type, col_col):

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

    # ####################
    # ##### PLOTTING #####
    # ####################
    def plot_map_selected_units(self, scenario='', save=True):
        """

        """
        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                gdf_boundaries_regions = self.gdf_boundaries_regions[['any', scenario]]
                gdf_points_units = self.gdf_points_units[['any', scenario]]
                gdf_lines_transfer = self.gdf_lines_transfer[['any', scenario]]
            except KeyError:
                # The scenario isn't in the Data.
                # TO DO - proper error message here
                print('oh no')
                raise KeyError('Scenario is missing.') from None

            # Remove the excess column heading:
            gdf_boundaries_regions = gdf_boundaries_regions.droplevel(0, axis='columns')
            gdf_points_units = gdf_points_units.droplevel(0, axis='columns')
            gdf_lines_transfer = gdf_lines_transfer.droplevel(0, axis='columns')
            # The geometry column is still defined with the excess
            # heading, so update which column is geometry:
            gdf_boundaries_regions = gdf_boundaries_regions.set_geometry('geometry')
            gdf_points_units = gdf_points_units.set_geometry('geometry')
            gdf_lines_transfer = gdf_lines_transfer.set_geometry('geometry')
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions
            gdf_points_units = self.gdf_points_units
            gdf_lines_transfer = self.gdf_lines_transfer

        fig, ax = plt.subplots(figsize=(10, 10))

        ax = maps.plot_map_selected_units(
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax
        )

        if save:
            dir_output = ''
            try:
                for d in self.setup.list_dir_output:
                    end = os.path.split(d)[-1]
                    if end == scenario:
                        dir_output = d
            except AttributeError:
                # Setup is not defined.
                pass

            try:
                file_name = self.setup.file_selected_units_map
            except AttributeError:
                # Setup is not defined.
                file_name = f'map_selected_units_{scenario}.jpg'

            path_to_file = os.path.join(dir_output, file_name)
            plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_map_catchment(self, scenario='', boundary_kwargs={}, save=True):
        """

        """
        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                gdf_boundaries_lsoa = self.gdf_boundaries_lsoa[['any', scenario]]
                gdf_boundaries_regions = self.gdf_boundaries_regions[['any', scenario]]
                gdf_points_units = self.gdf_points_units[['any', scenario]]
                gdf_lines_transfer = self.gdf_lines_transfer[['any', scenario]]
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
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_geometry('geometry')
            gdf_boundaries_regions = gdf_boundaries_regions.set_geometry('geometry')
            gdf_points_units = gdf_points_units.set_geometry('geometry')
            gdf_lines_transfer = gdf_lines_transfer.set_geometry('geometry')

        else:
            gdf_boundaries_lsoa = self.gdf_boundaries_lsoa
            gdf_boundaries_regions = self.gdf_boundaries_regions
            gdf_points_units = self.gdf_points_units
            gdf_lines_transfer = self.gdf_lines_transfer

            # Remove the excess column headings:
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.droplevel(
                1, axis='columns')

        lsoa_boundary_kwargs = {
            'column': 'postcode_nearest',
            'cmap': 'Blues',
            'edgecolor': 'face'
        }
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(scenario)

        ax = maps.plot_map_catchment(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            boundary_kwargs=lsoa_boundary_kwargs
        )

        if save:
            dir_output = ''
            try:
                for d in self.setup.list_dir_output:
                    end = os.path.split(d)[-1]
                    if end == scenario:
                        dir_output = d
            except AttributeError:
                # Setup is not defined.
                pass

            try:
                file_name = self.setup.file_catchment_map
            except AttributeError:
                # Setup is not defined.
                file_name = f'map_catchment_{scenario}.jpg'

            path_to_file = os.path.join(dir_output, file_name)
            plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()