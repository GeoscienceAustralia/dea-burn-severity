from .burn_severity_config import RuntimeBurnConfig
from .burn_severity_processing import BurnSeverityProcessor
from .database import InputDatabase


def cli() -> None:

    config = RuntimeBurnConfig(db_use_status_table=True, upload_to_s3=True)
    database = InputDatabase(config)

    polygons = database.load_filtered_polygons()

    if polygons is None or polygons.empty:
        print("No polygons loaded. Exiting.")
        return

    burn_processing = BurnSeverityProcessor(config)
    burn_processing.process_all_polygons(polygons)
