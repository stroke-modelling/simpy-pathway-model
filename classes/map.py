"""
Stick this in a new home eventually. Not a class yet.
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
    gdf_boundaries = geopandas.read_file(path_to_file)#, crs='EPSG:27700')
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
    try:
        long = df_units.Easting
        lat = df_units.Northing
    except AttributeError:
        long = df_units.long_x
        lat = df_units.lat_x
    df_units['geometry'] = geopandas.points_from_xy(long, lat)
    # Make a column of coordinates [x, y]:
    df_units['coords'] = df_units[['Easting', 'Northing']].values.tolist()

    # Convert to GeoDataFrame:
    gdf_units = geopandas.GeoDataFrame(
        df_units, geometry=df_units['geometry']#, crs="EPSG:4326"
    )
    return gdf_units


def import_transfer_unit_data(setup):
    # Import transfer stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_national_transfer_units
    path_to_file = os.path.join(dir_input, file_input)
    df_transfer = pd.read_csv(path_to_file)
    return df_transfer


def keep_only_selected_units(
        df_transfer, df_units, left_col, right_col, how='right'):
    # Shorten the transfer unit data to just selected units.
    df_transfer = pd.merge(
        df_transfer, df_units[right_col],
        left_on=left_col, right_on=right_col,
        how=how
    )
    return df_transfer


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
def draw_boundaries(ax, gdf_boundaries, column=None,
                    cmap='Blues', edgecolor='silver', facecolor=None):
    # Draw the main map with colours (choropleth):
    gdf_boundaries.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        facecolor=facecolor,
        column=column,
        cmap='Blues',
        edgecolor='silver',
        linewidth=0.5
        # edgecolor='face',   # Make LSOA boundry same colour as area
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

    ax = draw_boundaries(ax, gdf_boundaries, column=gdf_boundaries.index.name)
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
