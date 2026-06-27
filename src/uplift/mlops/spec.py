from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


# ============================ Custom Specs ======================================
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

# ============================ Optuna Specs ======================================

class OptunaIntegerSpec(BaseModel):
    type: Literal['int']
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False

class OptunaCategoricalSpec(BaseModel):
    type: Literal['categorical']
    name: str
    choices: list[Any]
    @model_validator(mode="after")
    def validate_choices(self):
        choice_type = type(self.choices[0])
        for choice in self.choices:
            if type(choice) != choice_type:
                raise ValueError(f"Choices must has all the same type. Found type {choice_type} and {type(choice)} in choices.")
        return self

class OptunaUniformSpec(BaseModel):
    type: Literal['uniform']
    name: str
    low: int
    high: int

OptunaParameterSpec = Annotated[
    OptunaIntegerSpec | OptunaCategoricalSpec | OptunaUniformSpec, Field(discriminator="type")
]
