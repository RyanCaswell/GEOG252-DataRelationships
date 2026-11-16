import pandas as pd
import numpy as np
import pyreadstat
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy import stats

def chi_square( df, col1, col2 ):
    data = df[ [ col1, col2 ] ].dropna()

    table = pd.crosstab( data[col1], data[col2] )

    if table.shape[0] < 2 or table.shape[1] < 2:
        return None

    chi2, p, dof, expected = stats.chi2_contingency(
        table, correction=False
    )

    n = table.to_numpy().sum()

    cramers_v = np.sqrt(
        chi2 / ( n * min( table.shape[0] - 1, table.shape[1] - 1 ) )
    )

    test = "Chi-square"
    statistic = chi2
    statistic_type = "Chi-square"

    if table.shape == ( 2, 2 ) and ( expected < 5 ).any():
        odds_ratio, p = stats.fisher_exact( table )
        test = "Fisher's exact"
        statistics = odds_ratio
        statistics_type = "Odds ratio"
    else:
        test = "Chi-Square"

    return {
        "Variable 1": col1,
        "Variable 2": col2,
        "Test": test,
        "Statistic": statistic,
        "Statistic Type": statistic_type,
        "P-Value": p,
        "Effect Size": cramers_v,
        "Effective Type": "Cramer's V",
        "N": n,
        "Small Expected Counts": ( expected < 5 ).any()
    }

def anova( df, col1, col2 ):
    # col1 is nominal, col2 is scale

    data = df[ [ col1, col2 ] ].dropna()
    data[col2] = pd.to_numeric( data[col2], errors="coerce" )
    data = data.dropna()

    groups = [
        group[ col2 ].to_numpy()
        for _, group in data.groupby( col1 )
    ]

    if len( groups ) < 2 or any( len( group ) == 0 for group in groups ):
        return None
    
    if data[col2].nunique() < 2:
        return None

    result = stats.f_oneway( *groups )

    grand_mean = data[ col2 ].mean()

    ss_between = sum(
        len( group ) * ( group.mean() - grand_mean ) ** 2
        for group in groups
    )


    ss_total = ( ( data[col2] - grand_mean ) ** 2 ).sum()

    eta_squared = ss_between / ss_total

    return {
        "Variable 1": col1,
        "Variable 2": col2,
        "Test": "ANOVA",
        "Statistic": result.statistic,
        "P-Value": result.pvalue,
        "Effect Size": eta_squared,
        "effect Type": "Eta-squared",
        "N": len( data )
    }

def pearson( df, col1, col2 ):
    data = df[ [ col1, col2, ] ].dropna()
    data = data.apply( pd.to_numeric, errors="coerce" ).dropna()

    if len( data ) < 3:
        return None

    if data[ col1 ].nunique() < 2 or data[ col2 ].nunique() < 2:
        return None

    result = stats.pearsonr( data[ col1 ], data[ col2 ] )

    return {
        "Variable 1": col1,
        "Variable 2": col2,
        "Test": "Pearson",
        "Statistic": result.statistic,
        "P-Value": result.pvalue,
        "Effect Size": result.statistic,
        "effect Type": "Absolute r",
        "N": len( data )
    }

def spearman( df, col1, col2 ):
    data = df[ [ col1, col2, ] ].dropna()
    data = data.apply( pd.to_numeric, errors="coerce" ).dropna()

    if len( data ) < 3:
        return None

    if data[ col1 ].nunique() < 2 or data[ col2 ].nunique() < 2:
        return None    

    result = stats.spearmanr( data[ col1 ], data[ col2 ] )

    return {
        "Variable 1": col1,
        "Variable 2": col2,
        "Test": "Spearman",
        "Statistic": result.statistic,
        "P-Value": result.pvalue,
        "Effect Size": result.statistic,
        "effect Type": "Absolute rho",
        "N": len( data )
    }



df, metadata = pyreadstat.read_sav( "data/recoded.sav")

other_cols = [ col for col in df.columns if col.endswith( "_other" ) ]

q2_cols = [ col for col in df.columns if col.startswith( "Q2_" ) ]

ignore_cols = q2_cols + other_cols + [
    "IPAddress",
    "p0001",
    "p0002",
    "p0003",
    "p0004",
    "p0005",
    "p0006",
    "p0007",
    "Q21",
    "Q22",
    "Q17",
    "Q3_CategoryVariableforUsualModestoPublicTransit",
]

independant_cols = [ 
    # "Q1",
    # "Q18",
    # "Q19",
    # "Q23",

    # "Q2_01_01",
    # "Q2_02_01",
    # "Q2_03_01",

    # "Q3_02",
    # "Q3_04",
    # "Q3_09",
    # "Q3_10",

    # "Q6_03",
    # "Q6_04",
    # "Q6_05",
    # "Q6_06",

    # "Q7_01",
    # "Q7_02",
    # "Q7_03",
    # "Ethnicity_Categorical",
    "micromobility_user",
 ]

all_cols = df.columns.tolist()
use_cols = [ col for col in all_cols if col not in ignore_cols and col not in independant_cols]

# use_cols = [
#     "Q1",
#     "Q6_01",
#     "Q6_02",
#     "Q6_03",
#     "Q6_04",
#     "Q6_05",
#     "Q6_06",
#     "Q6_07",
#     "Q6_08",
#     "Q6_09",
#     "Q6_10",
#     "Q6_11",

#     "Q7_01",
#     "Q7_02",
#     "Q7_03",
#     "Q7_04",
#     "Q7_05",
#     "Q7_06",
#     "Q7_07",
#     "Q7_08",
#     "Q7_09",
#     "Q7_10",

#     "Q14_01",
#     "Q14_02",
#     "Q14_03",
#     "Q14_04",
#     "Q14_05",
#     "Q14_06",

#     "Q16_01",
#     "Q16_02",
#     "Q16_03",
#     "Q16_04",
#     "Q16_05",
#     "Q16_06",
# ]

results = []

skipped = 0

# Get pairs of all 
for col1 in independant_cols:
    for col2 in use_cols:
        # Skip comparing matching columns
        if col1 == col2:
            continue

        types = []
        types.append( metadata.variable_measure.get( col1 ) )
        types.append( metadata.variable_measure.get( col2 ) )

        result = {}

        # Skip broken types
        if len( types ) != 2 or None in types:
            print( f"Types not found - {col1}, {col2}" )
            continue

        if "nominal" in types:
            if "scale" in types:
                # Nominal/Scale
                nominal_col = col1 if types[0] == "nominal" else col2
                scale_col = col1 if types[0] == "scale" else col2

                result = anova( df, nominal_col, scale_col )
            else:
                # Nominal/Nominal and Nominal/Ordinal
                result = chi_square( df, col1, col2 )

        elif "ordinal" in types:
            # Ordinal/Ordinal and Ordinal/Scale
            result = spearman( df, col1, col2 )

        elif "scale" in types:
            # Scale/Scale
            result = pearson( df, col1, col2 )

        else:
            print( f"No suitable statistic - {col1}, {col2}" )
            skipped += 1

        if result:
            results.append( result )
        else:
            skipped += 1

        processed = len( results )
        total = processed + skipped

        print( f"{processed}\t{skipped}\t{total}")

results_df = pd.DataFrame( results )

if not results_df.empty:
    results_df = results_df.sort_values( "Effect Size", ascending=False )

    results_df.to_csv( "data/results.csv", index=False )

    print_cols = [
        "Variable 1",
        "Variable 1 Description",
        "Variable 2",
        "Variable 2 Description",
        "P-Value",
        "Effect Size",
    ]

    significant = results_df[ results_df["P-Value"] < 0.05 ]

    for _, row in significant.head( 10 ).iterrows():
        col1 = row[ "Variable 1" ]
        col2 = row[ "Variable 2" ]

        desc1 = metadata.column_names_to_labels.get( col1 ) or col1
        desc2 = metadata.column_names_to_labels.get( col2 ) or col2

        print( f"\n{col1}: {desc1}" )
        print( f"{col2}: {desc2}" )
        print( f"P-Value: {row["P-Value"]:.5g}" )
        print( f"Effect Size: {row["Effect Size"]:.3f}" )