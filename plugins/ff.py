from enum import Enum, auto
import inspect
import math
import os
from typing import Any, Callable, Dict, List, Union, get_args, get_origin
from mirai import FriendMessage, GroupMessage, MessageEvent
from ..plugin import Plugin, autorun, instr
import json
from dataclasses import dataclass
import csv
from typing import TypeVar, Generic
import re
from collections.abc import Iterable

RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ff')

T = TypeVar('T')

class FFCsv(Generic[T]):
    def __init__(self, db: 'Db', table_name: str, enum_type: type) -> None:
        self.table_name = table_name
        self.db = db
        self.enum_type = enum_type

    def __getitem__(self, key):
        return self.d[key]

    def load(self):
        with open(os.path.join(RESOURCE_PATH, f'{self.table_name}.csv'), encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            self.key_names = next(reader)
            self.item_types = next(reader)
            d = {}
            for row in reader:
                d[row['key']] = FFCsvRow(self, row)
        self.d = d

    def values(self) -> List['FFCsvRow[T]']:
        return self.d.values()

    def keys(self):
        return self.d.keys()

    def items(self):
        return self.d.items()

    def find_by(self, key: Union[List[Union[T, str, int]], Union[T, str, int]], value):
        for row in self.d.values():
            if not isinstance(key, Iterable):
                key = [key]
            for k in key:
                if row[k] == value:
                    return row

@dataclass
class FFCsvRow(Generic[T]):
    table: FFCsv
    row: Dict

    def __getitem__(self, key):
        if type(key) is int:
            key = str(key)
        if isinstance(key, Enum):
            if not isinstance(key, self.table.enum_type):
                raise RuntimeError(f'{self.table.table_name} not supported key {key}')
            key = key.value if type(key.value) is str else key.name
        if key in self.table.key_names.values():
            key = list(self.table.key_names.keys())[list(self.table.key_names.values()).index(key)]
        if self.table.item_types[key] in self.table.db.tables and Fk.match(self.table.item_types[key], self.row[key]):
            return Fk(self.table.db, self.row[key])
        if re.match('u?int\\d+|(byte$)', self.table.item_types[key]) is not None:
            return int(self.row[key])
        if re.match('bit&\\d+', self.table.item_types[key]) is not None:
            return bool(self.row[key])
        return self.row[key]

    @property
    def key(self):
        return self.row['key']

    def items(self):
        return self.row.items()

class Fk:
    db: 'Db'
    fk: str

    def __init__(self, db, fk) -> None:
        self.db = db
        self.fk = fk

    def query(self):
        table, item_level_fk = self.fk.split('#')
        return self.db[table][item_level_fk]

    @staticmethod
    def match(table: str, val: str) -> bool:
        return re.match(f'{table}#\d+', val) is not None

class GatheringItemKey(Enum):
    Item = auto()
    GatheringItemLevel = auto()

class SpearfishingItemKey(Enum):
    Item = auto()
    GatheringItemLevel = auto()

class GatheringPointBaseKey(Enum):
    GatheringType = auto()

class GatheringItemLevelConvertTableKey(Enum):
    GatheringItemLevel = auto()

class RecipeKey(Enum):
    ItemResult = "Item{Result}"
    CraftType = auto()
    RecipeLevelTable = auto()

class RecipeLevelTableKey(Enum):
    ClassJobLevel = auto()

class ItemKey(Enum):
    ...

# LeveRewardItemGroup
# LeveRewardItem

class Db():
    gathering_item: FFCsv[GatheringItemKey]
    spearfishing_item: FFCsv[SpearfishingItemKey]
    gathering_point_base: FFCsv[GatheringPointBaseKey]
    gathering_item_level_convert_table: FFCsv[GatheringItemLevelConvertTableKey]
    recipe: FFCsv[RecipeKey]
    recipe_level_table: FFCsv[RecipeLevelTableKey]
    item: FFCsv[ItemKey]

    def __init__(self) -> None:
        self.tables = {}
        self.register_table(*[(self.to_upper_case(name),get_args(t)[0]) for name, t in self.__class__.__annotations__.items() if get_origin(t) is FFCsv])

    def register_table(self, *table_names):
        for table_name, enum_type in table_names:
            table = FFCsv(self, table_name, enum_type)
            table.load()
            self.tables[table_name] = table

    def __getattr__(self, name) -> FFCsv:
        return self.tables[self.to_upper_case(name)]

    def __getitem__(self, name) -> FFCsv:
        return self.tables[self.to_upper_case(name)]

    @staticmethod
    def to_upper_case(s: str):
        res = ''.join(ele[0].upper() + ele[1:] for ele in s.split('_'))
        return res

    def get_gathering_item(self, name: str):
        result = GatheringItem()
        result.name = name
        point = self.gathering_point_base.find_by(range(2, 10), name)
        if point is None:
            return
        point_method = point[GatheringPointBaseKey.GatheringType]
        result.job.job = GatheringItem.job_from_method(point_method)
        g_item = self.gathering_item.find_by(GatheringItemKey.Item, name)
        s_item = self.spearfishing_item.find_by(SpearfishingItemKey.Item, name)
        level_row = None
        if g_item is not None:
            level_row = g_item[GatheringItemKey.GatheringItemLevel].query()
        if s_item is not None:
            level_row = s_item[SpearfishingItemKey.GatheringItemLevel].query()
        if level_row is not None:
            result.job.level = level_row[GatheringItemLevelConvertTableKey.GatheringItemLevel]
        return result

db = Db()

class Job(Enum):
    木工 = auto()
    锻冶 = auto()
    铸甲 = auto()
    雕金 = auto()
    制革 = auto()
    裁缝 = auto()
    炼金 = auto()
    烹调 = auto()

    采矿 = auto()
    园艺 = auto()
    捕鱼 = auto()

    Unknown = auto()

class RequiredJob():
    job: Job
    level: int

    def __init__(self) -> None:
        self.job = Job.Unknown
        self.level = 0

    def __str__(self) -> str:
        if self.job is Job.Unknown:
            return '?'
        return f'{self.job.name}:{self.level}'

class Recipe():
    def __init__(self, name: str, result_count: int = 1) -> None:
        self.name = name
        self.jobs: List[RequiredJob] = []
        self.materials: List['Material'] = []
        self.result_count = result_count

    def parse(self, count: int, *decos: Callable[[str], Any]): # 需要的成品个数
        times = math.ceil(count / self.result_count) # 制作次数
        s = []

        job = []
        if len(self.jobs) > 0:
            job = self.jobs
        else:
            item = db.get_gathering_item(self.name)
            if item is not None:
                job = [item.job]

        s.append(f'{self.name} x {times * self.result_count} [{" ".join([str(j) for j in job])}]{" ".join([""] + [str(d()) for d in decos])}')

        sub = []
        for m in self.materials:
            sub.extend([f'  {r}' for r in m.recipe.parse(times * m.count)])
        s.extend(sub)
        return s

    @classmethod
    def build(self, name: str):
        r = Recipe(name)
        matched = False
        for recipe in db.recipe.values():
            if recipe[RecipeKey.ItemResult] == name:
                job = recipe[RecipeKey.CraftType]
                rj = RequiredJob()
                rj.job = Job[job]
                rj.level = recipe[RecipeKey.RecipeLevelTable].query()[RecipeLevelTableKey.ClassJobLevel]
                r.jobs.append(rj)
                if not matched:
                    matched = True
                    for i in range(8):
                        item_name = recipe[f'Item{{Ingredient}}[{i}]']
                        item_amount = recipe[f'Amount{{Ingredient}}[{i}]']
                        if item_amount > 0:
                            r.materials.append(Material(self.build(item_name), item_amount))
        return r

class Material():
    def __init__(self, recipe: Recipe, count: int) -> None:
        self.recipe = recipe
        self.count = count

@dataclass
class GatheringItem():
    name: str
    job: RequiredJob

    def __init__(self) -> None:
        self.job = RequiredJob()

    @staticmethod 
    def job_from_method(method: str) -> Job:
        if method in ('采掘', '碎石'): return Job.采矿
        if method in ('采伐', '割草'): return Job.园艺
        if method in ('●銛',): return Job.捕鱼
        return Job.Unknown

class FF(Plugin):
    def __init__(self) -> None:
        super().__init__('ff')

    @instr('配方')
    async def recipe(self, name: str, count: int = 1):
        return ['\n'.join((Recipe.build(name)).parse(count))]
