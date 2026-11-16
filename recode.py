import pandas as pd
import pyreadstat

# Load dataframe and metadata
df, metadata = pyreadstat.read_sav( "data/original.sav")

# Remove "other" columns from dataframe and metadata
remove_cols = [ col for col in df.columns if col.endswith( "_other" ) ]

keep_indexes = [ i for i, col in enumerate( df.columns ) if col not in remove_cols ]

df = df.drop( columns=remove_cols)

metadata.column_labels = [ metadata.column_labels[i] for i in keep_indexes ]

for key in remove_cols:
    metadata.variable_value_labels.pop( key, None )
    metadata.variable_measure.pop( key, None )

def zero_columns( df, q_num ):
    # Used for multiple answer questions to ensure answers are 0 or 1
    cols = [
        col for col in df.columns
        if col.startswith( f"Q{q_num}_" )
    ]

    answered = df[ cols ].notna().any( axis=1 )

    df.loc[ answered, cols ] = df.loc[ answered, cols ].fillna( 0 )

def trim_recode( df, q_num, keep):
    if q_num != 2:
        cols = [ col for col in df.columns if col.startswith( f"Q{q_num}_"  ) ]
    else:
        cols = [ col for col in df.columns if col.startswith( f"Q{q_num}_" ) and col.endswith( "_02" ) ]

    for col in cols:
        if keep == 0:
            df.loc[ df[ col ] == 0, col ] = pd.NA
        else:
            df.loc[ df[ col ] > keep, col ] = pd.NA

multi = [ 3, 6, 7, 13, 14, 15, 16 ]

trim = [
    ( 2, 3 ),
    ( 4, 5 ),
    ( 5, 2  ),
    ( 8, 2 ),
    ( 9, 3 ),
    ( 10, 5 ),
    ( 11, 5 ),
    ( 12, 5 ),
    ( 19, 0 ),
    ( 20, 0 ),
    ( 27, 2 ),
    ( 28, 8 ),
]

usage_cols = [
    "Q2_01_01",  # E-bike
    "Q2_02_01",  # Conventional bicycle
    "Q2_03_01",  # Kick e-scooter
    "Q2_04_01",  # Kick scooter
    "Q2_05_01",  # Seated limited speed electric scooter
    "Q2_07_01",  # E-skateboard
    "Q2_08_01",  # Skateboard
    "Q2_09_01",  # Single axis electical mobility device
]

user_values = [ 4, 5, 6 ]

for q_num in multi:
    zero_columns( df, q_num )

for q_num, keep in trim:
    trim_recode( df, q_num, keep )

micro_user = pd.DataFrame( { "micromobility_user": ( df[ usage_cols].isin( user_values ).any( axis=1 ).astype( int ) ) } )

df = pd.concat( [df, micro_user ], axis=1 )

metadata.column_names_to_labels["micromobility_user"] = ( "Current micromobility use (at least monthly)" )
metadata.variable_measure["micromobility_user"] = "nominal"
metadata.variable_value_labels["micromobility_user"] = { 0: "Non-user", 1: "User" }

pyreadstat.write_sav(
    df,
    "data/recoded.sav",
    column_labels=metadata.column_names_to_labels,
    variable_value_labels=metadata.variable_value_labels,
    variable_measure=metadata.variable_measure
)