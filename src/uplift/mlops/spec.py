import pdb
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChoiceSpec(BaseModel):
    type: Literal["choice"]
    values: list[Any]


class IntRangeSpec(BaseModel):
    type: Literal["int_range"]
    min: int
    max: int
    step: int = 1

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min > self.max:
            raise ValueError("must have min < max")
        return self


class FloatRangeSpec(BaseModel):
    type: Literal["float_range"]
    min: float
    max: float

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min > self.max:
            raise ValueError("must have min < max")
        return self


ParameterSpec = Annotated[
    ChoiceSpec | IntRangeSpec | FloatRangeSpec, Field(discriminator="type")
]


class SearchSpaceConfig(BaseModel):
    hyperparameters: dict[str, ParameterSpec]


if __name__ == "__main__":
    data = {"type": "int_range", "min": 3, "max": 21, "step": 2}

    data = IntRangeSpec(**data)

    hp = {
        "param1": {"type": "int_range", "min": 1, "max": 9, "step": 2},
        "param2": {"type": "choice", "values": ["a", "b", "c"]},
    }

    config = SearchSpaceConfig.model_validate({"hyperparameters": hp})
    pdb.set_trace()
