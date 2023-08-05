from datetime import timedelta
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.types import Float32, Float64, Int64, Int32


# user entity
user_entity = Entity(name="user", join_keys=["userid"])
user_source = FileSource(
    path="/home/hs/user_entity.parquet",
    timestamp_field="timestamp_field",
    created_timestamp_column="created_timestamp_column",
    name="user_source"
)
user_fv = FeatureView(
    name="user_fv",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="gender", dtype=Int32),
        Field(name="age", dtype=Int32),
        Field(name="occupation", dtype=Int32),
        Field(name="user_g1_imp", dtype=Float32),
        Field(name="user_g1_clk", dtype=Float32),
        Field(name="user_g2_imp", dtype=Float32),
        Field(name="user_g2_clk", dtype=Float32),
        Field(name="user_g3_imp", dtype=Float32),
        Field(name="user_g3_clk", dtype=Float32),
        Field(name="user_g4_imp", dtype=Float32),
        Field(name="user_g4_clk", dtype=Float32),
        Field(name="user_g5_imp", dtype=Float32),
        Field(name="user_g5_clk", dtype=Float32),
        Field(name="user_g6_imp", dtype=Float32),
        Field(name="user_g6_clk", dtype=Float32),
        Field(name="user_g7_imp", dtype=Float32),
        Field(name="user_g7_clk", dtype=Float32),
        Field(name="user_g8_imp", dtype=Float32),
        Field(name="user_g8_clk", dtype=Float32),
        Field(name="user_g9_imp", dtype=Float32),
        Field(name="user_g9_clk", dtype=Float32),
        Field(name="user_g10_imp", dtype=Float32),
        Field(name="user_g10_clk", dtype=Float32),
        Field(name="user_g11_imp", dtype=Float32),
        Field(name="user_g11_clk", dtype=Float32),
        Field(name="user_g12_imp", dtype=Float32),
        Field(name="user_g12_clk", dtype=Float32),
        Field(name="user_g13_imp", dtype=Float32),
        Field(name="user_g13_clk", dtype=Float32),
        Field(name="user_g14_imp", dtype=Float32),
        Field(name="user_g14_clk", dtype=Float32),
        Field(name="user_g15_imp", dtype=Float32),
        Field(name="user_g15_clk", dtype=Float32),
        Field(name="user_g16_imp", dtype=Float32),
        Field(name="user_g16_clk", dtype=Float32),
        Field(name="user_g17_imp", dtype=Float32),
        Field(name="user_g17_clk", dtype=Float32),
        Field(name="user_g18_imp", dtype=Float32),
        Field(name="user_g18_clk", dtype=Float32),
    ],
    online=True,
    source=user_source,
)

# item entity
item_entity = Entity(name="item", join_keys=["itemid"])
item_source = FileSource(
    path="/home/hs/item_entity.parquet",
    timestamp_field="timestamp_field",
    created_timestamp_column="created_timestamp_column",
    name="item_source"
)
item_fv = FeatureView(
    name="item_fv",
    entities=[item_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="item_g1", dtype=Float32),
        Field(name="item_g2", dtype=Float32),
        Field(name="item_g3", dtype=Float32),
        Field(name="item_g4", dtype=Float32),
        Field(name="item_g5", dtype=Float32),
        Field(name="item_g6", dtype=Float32),
        Field(name="item_g7", dtype=Float32),
        Field(name="item_g8", dtype=Float32),
        Field(name="item_g9", dtype=Float32),
        Field(name="item_g10", dtype=Float32),
        Field(name="item_g11", dtype=Float32),
        Field(name="item_g12", dtype=Float32),
        Field(name="item_g13", dtype=Float32),
        Field(name="item_g14", dtype=Float32),
        Field(name="item_g15", dtype=Float32),
        Field(name="item_g16", dtype=Float32),
        Field(name="item_g17", dtype=Float32),
        Field(name="item_g18", dtype=Float32),
    ],
    online=True,
    source=item_source,
)

# feature
ml_user = FeatureService(
    name="ml_user",
    features=[user_fv]
)
ml_item = FeatureService(
    name="ml_item",
    features=[item_fv]
)
