"""Small JSON codec for the public dataclass schemas.

Nested construction and validation belong to each schema's ``__post_init__``;
this module only handles JSON representation and schema descriptions.
"""
import json
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _type_schema(annotation: Any) -> dict:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return {'type': 'string', 'enum': list(args)}
    if origin is Union:
        return {'anyOf': [_type_schema(item) for item in args]}
    if origin in (list, List):
        return {'type': 'array', 'items': _type_schema(args[0])}
    if origin in (dict, Dict):
        return {'type': 'object', 'additionalProperties': _type_schema(args[1])}
    if annotation is datetime:
        return {'type': 'string', 'format': 'date-time'}
    if isinstance(annotation, type) and issubclass(annotation, SerialisableBaseModel):
        return annotation.schema()
    return {'type': {float: 'number', int: 'integer', str: 'string', bool: 'boolean'}.get(annotation, 'object')}


class SerialisableBaseModel:
    """JSON compatibility methods shared by explicitly validated dataclasses."""

    def dict(self) -> dict:
        return _encode(self)

    def json(self, **kwargs) -> str:
        return json.dumps(self.dict(), **kwargs)

    @classmethod
    def parse_obj(cls, value: dict):
        return cls(**value)

    @classmethod
    def parse_raw(cls, value: Union[str, bytes]):
        return cls.parse_obj(json.loads(value))

    @classmethod
    def schema(cls) -> dict:
        hints = get_type_hints(cls)
        properties, required = {}, []
        for field in fields(cls):
            description = _type_schema(hints[field.name])
            description.update(field.metadata)
            if field.default is not MISSING:
                description['default'] = _encode(field.default)
            elif field.default_factory is MISSING:
                required.append(field.name)
            properties[field.name] = description
        result = {'title': cls.__name__, 'type': 'object', 'properties': properties}
        if required:
            result['required'] = required
        return result

    @classmethod
    def schema_json(cls, **kwargs) -> str:
        return json.dumps(cls.schema(), **kwargs)


class HashableSerialisableBaseModel(SerialisableBaseModel):
    def __hash__(self):
        return hash(self.json(sort_keys=True))


def example_from_schema(model: Type[SerialisableBaseModel]) -> Dict[str, Any]:
    return {field.name: field.metadata['example'] for field in fields(model) if 'example' in field.metadata}


_T = TypeVar('_T')


def build_example(model: Type[_T]) -> _T:
    return model(**example_from_schema(model))


def apply_validators(value: Any, validators: List[Callable]) -> Any:
    for validator in validators:
        value = validator(value)
    return value
