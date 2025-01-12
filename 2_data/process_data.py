import pandas
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine


def load_data(path_to_input_data: Path) -> pandas.DataFrame:
    """
    Load data (e.g. messages, categories) from CSV file into pandas dataframe

    :param path_to_input_data: path to csv file
    :return: load as pandas dataframe
    """
    input_df = pd.read_csv(path_to_input_data)
    return input_df


def clean_and_transform_data(messages_df: pandas.DataFrame, categories_df: pandas.DataFrame) -> pandas.DataFrame:
    """
    transforming and cleaning the dataframe (df).

    :param messages_df: input dataframe for disaster messages
    :param categories_df: input df for categories
    :return: cleaned and transformed df
    """

    # merge both loaded dataframes
    df = pd.merge(messages_df, categories_df, on='id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    # up to the second to last character of each string with slicing
    category_colnames = [cat[:-2] for cat in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories' column from `df`
    df.drop(columns='categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    # check count of rows are identical
    if df.shape[0] == categories.shape[0]:
        df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_to_database(df: pandas.DataFrame, table_name: str) -> None:
    """
    Save dataframe (df) to sql database (db)

    :param df: dataframe to be saved
    :param table_name: name of table in sql db
    :return: None
    """
    engine = create_engine('sqlite:///Messages.db')
    df.to_sql(str(table_name), engine, index=False)


if __name__ == '__main__':
    # extract from source
    input_messages_df = load_data(Path("messages.csv"))
    input_categories_df = load_data(Path("categories.csv"))

    # clean and transform
    clean_transformed_df = clean_and_transform_data(messages_df=input_messages_df,
                                                    categories_df=input_categories_df)
    # load / save to database
    save_to_database(df=clean_transformed_df, table_name="ETL_Messages")
