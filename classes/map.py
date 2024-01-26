"""
Stick this in a new home eventually. Not a class yet.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).
"""
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.


# ##########################
# ##### DATA WRANGLING #####
# ##########################
def import_geojson(setup, col):
    # Select geojson file based on input "col":
    geojson_file_dict = {
        'LSOA11NM': setup.file_geojson_lsoa,
        'CCG19NM': setup.file_geojson_ccg,
        'ICB22NM': setup.file_geojson_icb,
        'LAD17NM': setup.file_geojson_lad,
        'STP19NM': setup.file_geojson_stp,
        'LHB20NM': setup.file_geojson_lhb,
        'SCN17NM': setup.file_geojson_scn,
        'RGN11NM': setup.file_geojson_rgn,
    }
    # Import region file:
    dir_input = setup.dir_data_geojson
    file_input = geojson_file_dict[col]
    path_to_file = os.path.join(dir_input, file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)
    # If crs is given in the file, geopandas automatically
    # pulls it through.
    # Convert to National Grid coordinates:
    if gdf_boundaries.crs != 'EPSG:27700':
        gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
    return gdf_boundaries


def import_selected_stroke_units(setup):
    # Import selected stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_selected_stroke_units
    path_to_file = os.path.join(dir_input, file_input)
    df_units = pd.read_csv(path_to_file)

    # Create coordinates:
    # Current setup means sometimes these columns have different names.
    # TO DO - fix that please! ---------------------------------------------------
    x = df_units['Easting']
    y = df_units['Northing']
    xy = df_units[['Easting', 'Northing']]
    crs = 'EPSG:27700'

    df_units['geometry'] = geopandas.points_from_xy(x, y)
    # Make a column of coordinates [x, y]:
    df_units['coords'] = xy.values.tolist()

    # Convert to GeoDataFrame:
    gdf_units = geopandas.GeoDataFrame(
        df_units, geometry=df_units['geometry'], crs=crs
    )
    if crs != 'EPSG:27700':
        gdf_units = gdf_units.to_crs('EPSG:27700')
    return gdf_units


def import_transfer_unit_data(setup):
    # Import transfer stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_national_transfer_units
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def import_selected_lsoa(setup):
    # Import selected stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_selected_lsoas
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def import_lsoa_travel_data(setup):
    # Import selected stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_national_lsoa_travel
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def keep_only_selected_units(
        df, df_units, left_col, right_col, how='right'):
    # Shorten the transfer unit data to just selected units.
    df = pd.merge(
        df, df_units[right_col],
        left_on=left_col, right_on=right_col,
        how=how
    )
    return df


def copy_coords_selected_units(df_transfer, df_units):
    # Shorten the transfer unit data to just selected units.
    # Copy over the coordinates of each stroke unit.
    df_transfer = pd.merge(
        df_transfer, df_units[['Postcode', 'coords']],
        left_on='from_postcode', right_on='Postcode',
        how='right'
    )
    df_transfer = df_transfer.rename(columns={'coords': 'unit_coords'})
    # Now copy over the coordinates of each MT transfer unit.
    df_transfer = pd.merge(
        df_transfer, df_units[['Postcode', 'coords']],
        left_on='name_nearest_MT', right_on='Postcode',
        how='left'
    )
    df_transfer = df_transfer.rename(columns={'coords': 'transfer_coords'})
    return df_transfer


def create_lines_from_coords(df_transfer):
    # Create lines by connecting each unit to its transfer unit:
    df_transfer['line_coords'] = df_transfer[
        ['unit_coords', 'transfer_coords']].values.tolist()
    # Convert line coords to LineString objects:
    df_transfer['line_geometry'] = [
        LineString(coords) for coords in df_transfer['line_coords']]

    # Convert to GeoDataFrame:
    gdf_transfer = geopandas.GeoDataFrame(
        df_transfer, geometry=df_transfer['line_geometry']#, crs="EPSG:4326"
    )
    return gdf_transfer


# ####################
# ##### PLOTTING #####
# ####################
def draw_boundaries(ax, gdf_boundaries, **kwargs):
    # Draw the main map with colours (choropleth):
    gdf_boundaries.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        **kwargs
        )
    return ax


def scatter_units(ax, gdf_units):
    # Scatter marker for each hospital:
    gdf_units.plot(
        ax=ax,
        edgecolor='k',
        facecolor='w',
        markersize=50,
        marker='o',
        zorder=2
        )

    # Scatter marker star for MT/MSU units:
    mask = gdf_units['Use_MT'] == 1
    MSU = gdf_units[mask]
    MSU.plot(
        ax=ax,
        edgecolor='k',
        facecolor='y',
        markersize=300,
        marker='*',
        zorder=2
        )
    return ax


def plot_lines_between_units(ax, gdf_transfer):
    # Draw a line connecting each unit to its MT unit.
    gdf_transfer.plot(
        ax=ax,
        edgecolor='k',
        linestyle='-',
        linewidth=3,
        zorder=1  # Place it beneath the scatter markers,
    )
    return ax


def annotate_unit_labels(ax, gdf_units):
    # Add labels
    # Define "z" to shorten following "for" line:
    z = zip(
        gdf_units.geometry.x,
        gdf_units.geometry.y,
        gdf_units.Hospital_name
        )
    for x, y, label in z:
        # Edit the label to put a space in the postcode when displayed:
        label = f'{label[:-3]} {label[-3:]}'
        # Place the label slightly offset from the
        # exact hospital coordinates (x, y).
        ax.annotate(
            label, xy=(x, y), xytext=(8, 8),
            textcoords="offset points",
            # backgroundcolor="w",
            bbox=dict(facecolor='w', edgecolor='k'),
            fontsize=8
            )
    return ax


# ######################
# ##### MAIN PLOTS #####
# ######################
def plot_map_selected_units(setup, col='ICB22NM'):
    """
    WIP
    """
    # Import data:
    gdf_boundaries = import_geojson(setup, col)
    gdf_units = import_selected_stroke_units(setup)
    df_transfer = import_transfer_unit_data(setup)

    df_transfer = keep_only_selected_units(
        df_transfer, gdf_units,
        left_col='from_postcode', right_col='Postcode')
    df_transfer = copy_coords_selected_units(df_transfer, gdf_units)

    gdf_transfer = create_lines_from_coords(df_transfer)

    gdf_boundaries = keep_only_selected_units(
        gdf_boundaries, gdf_units, left_col=col, right_col=col)

    # Plot the map.
    fig, ax = plt.subplots(figsize=(10, 10))  # Make max dimensions XxY inch

    ax = draw_boundaries(
        ax, gdf_boundaries,
        column=gdf_boundaries.index.name,
        cmap='Blues', edgecolor='silver', linewidth=0.5)
    ax = scatter_units(ax, gdf_units)
    ax = plot_lines_between_units(ax, gdf_transfer)
    ax = annotate_unit_labels(ax, gdf_units)

    ax.set_axis_off()  # Turn off axis line and numbers

    # Save output to output folder.
    dir_output = setup.dir_output
    file_name = setup.file_selected_units_map
    path_to_file = os.path.join(dir_output, file_name)
    plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_map_drip_ship(setup, col='ICB22NM'):
    """
    WIP
    """
    # ----- Stroke unit setup -----
    gdf_units = import_selected_stroke_units(setup)

    # Find MT transfer units for plotting lines between units:
    df_transfer = import_transfer_unit_data(setup)
    df_transfer = keep_only_selected_units(
        df_transfer, gdf_units,
        left_col='from_postcode', right_col='Postcode')
    df_transfer = copy_coords_selected_units(df_transfer, gdf_units)
    gdf_transfer = create_lines_from_coords(df_transfer)

    # Find regional boundaries for reference on the map:
    gdf_boundaries = import_geojson(setup, col)
    gdf_boundaries = keep_only_selected_units(
        gdf_boundaries, gdf_units, left_col=col, right_col=col)

    # ----- LSOA setup -----
    df_lsoa = import_selected_lsoa(setup)

    # Find LSOA boundaries:
    gdf_boundaries_lsoa = import_geojson(setup, 'LSOA11NM')
    gdf_boundaries_lsoa = keep_only_selected_units(
        gdf_boundaries_lsoa,
        df_lsoa, left_col='LSOA11CD', right_col='LSOA11CD')

    # Match LSOA with its chosen stroke unit.
    df_lsoa_travel = import_lsoa_travel_data(setup)
    df_lsoa_travel = keep_only_selected_units(
        df_lsoa_travel, df_lsoa, left_col='LSOA11CD', right_col='LSOA11CD')
    cols_to_keep = [
        'LSOA11CD', 'postcode_nearest_IVT',
        'postcode_nearest_MT', 'postcode_nearest_MSU'
        ]
    gdf_boundaries_lsoa = pd.merge(
        gdf_boundaries_lsoa,
        df_lsoa_travel[cols_to_keep],
        left_on='LSOA11CD', right_on='LSOA11CD',
    )

    # ----- Plotting setup -----
    data_dicts = {
        'Drip & Ship': {
            'file': setup.file_drip_ship_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_IVT',
                'cmap': 'Blues',
                'edgecolor': 'face'
                }
            },
        'Mothership': {
            'file': setup.file_mothership_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_MT',
                'cmap': 'Blues',
                'edgecolor': 'face'
                }
            },
        'MSU': {
            'file': setup.file_msu_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_MSU',
                'cmap': 'Blues',
                'edgecolor': 'face'
                }
            },
    }

    # ----- Actual plotting -----
    for model_type, data_dict in zip(data_dicts.keys(), data_dicts.values()):
        # Plot the map.
        fig, ax = plt.subplots(figsize=(10, 10))  # Make max dimensions XxY inch
        ax.set_title(model_type)

        ax = draw_boundaries(
            ax, gdf_boundaries_lsoa,
            **data_dict['boundary_kwargs']
            )
        ax = draw_boundaries(
            ax, gdf_boundaries,
            facecolor='none', edgecolor='k', linewidth=0.5
            )
        ax = scatter_units(ax, gdf_units)
        ax = plot_lines_between_units(ax, gdf_transfer)
        ax = annotate_unit_labels(ax, gdf_units)

        ax.set_axis_off()  # Turn off axis line and numbers

        # Save output to output folder.
        dir_output = setup.dir_output
        file_name = data_dict['file']
        path_to_file = os.path.join(dir_output, file_name)
        plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
        plt.close()
