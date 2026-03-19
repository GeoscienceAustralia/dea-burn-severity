import os
import geopandas as gpd
import dea_burn_severity.burn_severity_config as burn_config
from dea_burn_severity.database import InputDatabase


def prep_outputs() -> None:
    # TODO this needs to be configurable for the notebook
    os.makedirs(burn_config.output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {burn_config.output_dir}")


def process_all_polygons(polygons: gpd.GeoDataFrame) -> None:
    for idx, entry in polygons[:100].iterrows():
        # Construct the output name based on the attributes.
        # TODO there will be dupes this way, everything can be repeated except the last thing
        # TODO maybe put the idx/rowid in the thing always?
        fire_name = entry["fire_name"] or entry["fire_id"] or f"fire_{idx}"
        print(fire_name)
        
        # TODO finish the output path setup on line 470ish of cli.py
        
        # ...
        
        map_burn_severity(entry)
        
        # ...
        
        # ... write result to db and anywhere else it needs to be

# for fire in list...
def map_burn_severity() -> None:
    pass


def write_output() -> None:
    # write cog
    # write polyg on
    # write to log
    pass
