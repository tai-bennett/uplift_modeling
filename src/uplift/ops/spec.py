from pydantic import BaseModel, Field, model_validator
from typing import Literal
import pdb

class IntRangeSpec(BaseModel):
    type: Literal['int_range']
    min: int
    max: int
    step: int = 1

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min > self.max:
            raise ValueError("must have min < max")
        return self

class FloatRangeSpec(BaseModel):
    type: Literal['float_range']
    min: float
    max: float

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min > self.max:
            raise ValueError("must have min < max")
        return self

class CategoricalSpec(BaseModel):
    type: Literal["categorical"]
    values: list

class FloatConstantSpec(BaseModel):
    type: Literal['float_constant']
    value: float

class IntConstantSpec(BaseModel):
    type: Literal['int_constant']
    value: float

class StrConstantSpec(BaseModel):
    type: Literal['str_constant']
    value: str

if __name__ == "__main__":
    data = {
        "type": "int_range",
        "min": 3,
        "max": 21,
        "step": 2
        }

    data = IntRangeSpec(**data)
