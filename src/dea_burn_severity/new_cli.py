from dea_burn_severity.burn_severity_config import RuntimeBurnConfig
from dea_burn_severity.burn_severity_processing import BurnSeverityProcessor
from dea_burn_severity.database import InputDatabase


def cli() -> None:

    config = RuntimeBurnConfig()
    database = InputDatabase(config)
    polygons = database.load_filtered_polygons()

    if polygons is None or polygons.empty:
        print("No polygons loaded. Exiting.")
        return
    
    burn_processing = BurnSeverityProcessor(config)
    burn_processing.process_all_polygons(polygons)
