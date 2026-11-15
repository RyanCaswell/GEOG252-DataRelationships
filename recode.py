import pandas as pd
import pyreadstat

df, metadata = pyreadstat.read_sav( "data/original.sav")

def zero_columns( df, q_num ):
    # Used for multiple answer questions to ensure answers are 0 or 1
    cols = [
        col for col in df.columns
        if col.startswith( f"Q{q_num}" ) and not col.endswith( "other" )
    ]

    answered = df[ cols ].notna().any( axis=1 )

    df.loc[ answered, cols ] = df.loc[ answered, cols ].fillna( 0 )

def trim_recode( df, q_num, keep):
    if q_num != 2:
        cols = [ col for col in df.columns if col.startswith( f"Q{q_num}_"  ) and not col.endswith( "_other" ) ]
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

for q_num in multi:
    zero_columns( df, q_num )

for q_num, keep in trim:
    trim_recode( df, q_num, keep )

pyreadstat.write_sav(
    df,
    "data/recoded.sav",
    column_labels=metadata.column_labels,
    variable_value_labels=metadata.variable_value_labels,
    variable_measure=metadata.variable_measure
)