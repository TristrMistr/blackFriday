import pandas as pd
from src.cleaning import convert_to_type
import pytest

@pytest.fixture
def example_df():
    data = {"int_column": [1,2,3],
            "float_column": [1.1,2.2,3.3]}
        
    df = pd.DataFrame(data, columns=["int_column","float_column"])

    return df

def test_convert_to_type_int_to_cat(example_df):
    assert(str(example_df.int_column.dtype) == "int64")
    result = convert_to_type(example_df, ["int_column"], "category")
    assert(str(result.int_column.dtype) == "category")

def test_convert_to_type_bad_option(example_df):
    with pytest.raises(TypeError) as error:
        result = convert_to_type(example_df, ["int_column"], "float")

    assert "Only supported options for dtype are \"integer\" and \"category\"" == str(error.value)