import pandas as pd

def missing_summary(df):
    """missing_summary 

    Calculates number and relative percentage of missing values in every column

    Args:
        df (pandas.DataFrame): Dataframe for which to calculate missing values  

    Returns:
        pandas.DataFrame: DataFrame with missing values and relevant percentages for each column
    """    
    missing_values = df.isna().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_summary = pd.DataFrame({
        "Missing Values": missing_values,
        "Percentage missing (%)": missing_percentage
    })

    return missing_summary

def dollar_to_int(df):
    """dollar_to_int 

    Turns all columns with entries in dollars into int type

    Args:
        df (pandas.DatFrame): Dataframe which to transform
    """    
    for col in df.columns:
        if str(df.loc[0,col]).startswith("$"):
            df[col] = df[col].apply(lambda x:int(x[1:]))

