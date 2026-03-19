import datacube

import dea_burn_severity.burn_severity_config as burn_config
import dea_burn_severity.burn_severity_processing as burn_processing
from dea_burn_severity.database import InputDatabase


def cli() -> None:

    dc = datacube.Datacube(app=burn_config.dc_app_name)

    database = InputDatabase()
    polygons = database.load_and_prepare_polygons()

    if polygons is None or polygons.empty:
        print("No polygons loaded. Exiting.")
        return

    print(polygons[:1])
    
    burn_processing.process_all_polygons(polygons)
