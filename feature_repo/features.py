from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32

# Entity: what identifies a record
# In our case, each reading is from a specific plant
solar_plant = Entity(
    name="plant_id",
    description="Solar plant identifier",
)

# Data source: where offline features live
weather_source = FileSource(
    path="data/feature_store/feature.parquet",
    timestamp_field="event_timestamp",
)

# Feature view: what features exist
weather_features = FeatureView(
    name="weather_features",
    entities=[solar_plant],
    ttl=timedelta(days=365),
    schema=[
        Field(name="IRRADIATION",        dtype=Float32),
        Field(name="MODULE_TEMPERATURE", dtype=Float32),
        Field(name="AMBIENT_TEMPERATURE",dtype=Float32),
        Field(name="HOUR",               dtype=Int32),
        Field(name="MONTH",              dtype=Int32),
        Field(name="DAY_OF_YEAR",        dtype=Int32),
        Field(name="HOUR_SIN",           dtype=Float32),
        Field(name="HOUR_COS",           dtype=Float32),
        Field(name="AC_POWER",           dtype=Float32),
        
    ],
    source=weather_source,
    online=True,
)