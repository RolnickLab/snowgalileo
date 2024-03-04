from datetime import date
from typing import Union


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")
