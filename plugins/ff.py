import asyncio
from enum import Enum, auto
import inspect
import math
from operator import iconcat
import os
import random
import stat
import sys
import time
from turtle import end_poly
from typing import Any, Awaitable, Callable, Dict, List, Set, Tuple, Union, get_args, get_origin
import urllib.parse

import aiohttp
from mirai import FriendMessage, GroupMessage, Image, MessageChain, MessageEvent, Plain, TempMessage
from mirai.models.entities import Friend

import mirai.models.message

from ..plugin import Plugin, autorun, instr
import json
from dataclasses import dataclass
import csv
from typing import TypeVar, Generic
import re
from collections.abc import Iterable

from mirai.models.message import Forward, ForwardMessageNode

from pyppeteer import launch

import numpy as np
import pandas as pd
from scipy.stats import kstest

import datetime

RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ff')
TMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

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

    def find_by(self, key: Union[List[Union[T, str, int]], Union[T, str, int]], value) -> 'FFCsvRow[T]':
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
        return self['key']

    @property
    def key_as_fk(self):
        return Fk.of(self.table, self.key)

    def items(self):
        return self.row.items()

class Fk:
    db: 'Db'
    fk: str

    def __init__(self, db, fk) -> None:
        self.db = db
        self.fk = fk

    def __eq__(self, __o: object) -> bool:
        return self.fk == __o

    def __ne__(self, __o: object) -> bool:
        return not(self == __o)

    def __hash__(self) -> int:
        return hash(self.fk)

    def query(self):
        table, item_level_fk = self.fk.split('#')
        return self.db[table][item_level_fk]

    @staticmethod
    def of(table: FFCsv, id: int):
        return Fk(table.db, f'{table.table_name}#{id}')
        ...

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
    AmountResult = "Amount{Result}"
    CraftType = auto()
    RecipeLevelTable = auto()

class RecipeLevelTableKey(Enum):
    ClassJobLevel = auto()

class ItemKey(Enum):
    Singular = auto()

class ShopItemGilKey(Enum):
    Item = auto()
    Gil = auto()

class Db():
    gathering_item: FFCsv[GatheringItemKey]
    spearfishing_item: FFCsv[SpearfishingItemKey]
    gathering_point_base: FFCsv[GatheringPointBaseKey]
    gathering_item_level_convert_table: FFCsv[GatheringItemLevelConvertTableKey]
    recipe: FFCsv[RecipeKey]
    recipe_level_table: FFCsv[RecipeLevelTableKey]
    item: FFCsv[ItemKey]
    shop_item_gil: FFCsv[ShopItemGilKey]

    recipe_index_result: Dict[str, List[FFCsvRow[RecipeKey]]]
    item_index_singular: Dict[str, List[FFCsvRow[ItemKey]]]
    shop_item_gil_index_item: Dict[str, List[FFCsvRow[ShopItemGilKey]]]
    gathering_item_index_item: Dict[str, List[FFCsvRow[GatheringItemKey]]]
    spearfishing_item_index_item: Dict[str, List[FFCsvRow[SpearfishingItemKey]]]

    def __init__(self) -> None:
        self.tables = {}
        self.register_table(*[(self.to_upper_case(name),get_args(t)[0]) for name, t in self.__class__.__annotations__.items() if get_origin(t) is FFCsv])
        self.recipe_index_result = self.create_index(self.recipe, RecipeKey.ItemResult)
        self.item_index_singular = self.create_index(self.item, ItemKey.Singular)
        self.shop_item_gil_index_item = self.create_index(self.shop_item_gil, ShopItemGilKey.Item)
        self.gathering_item_index_item = self.create_index(self.gathering_item, GatheringItemKey.Item)
        self.spearfishing_item_index_item = self.create_index(self.spearfishing_item, SpearfishingItemKey.Item)

    @staticmethod
    def create_index(table: FFCsv, field: Any):
        m: Dict[str, List[FFCsvRow]] = {}
        for rec in table.values():
            ikey = rec[field]
            if ikey not in m:
                m[ikey] = []
            m[ikey].append(rec)
        return m
            
        ...

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
        g_item = None
        if name in self.gathering_item_index_item:
            g_item = self.gathering_item_index_item[name][0]
        s_item = None
        if name in self.spearfishing_item_index_item:
            s_item = self.spearfishing_item_index_item[name][0]
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

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


class ItemQuality(Enum):
    NQ = auto()
    HQ = auto()

    def __str__(self) -> str:
        return 'HQ' if self is self.HQ else ''

class Item():
    crystals = [f'{x}之{y}' for x in ('火', '水', '土', '雷', '冰', '风') for y in ('碎晶', '水晶', '晶簇')]
    def __init__(self, name: str, quality: ItemQuality) -> None:
        self.name = name
        self.quality = quality
    
    def is_crystal(self):
        return self.name in self.crystals

    def __str__(self) -> str:
        return f'{self.name}{self.quality}'

    def __eq__(self, __o: 'Item') -> bool:
        return (self.name, self.quality) == (__o.name, __o.quality)

    def __hash__(self) -> int:
        return hash((self.name, self.quality))

    def __ne__(self, __o: object) -> bool:
        return not(self == __o)

    @staticmethod
    def parse(name_expr: str):
        name = name_expr
        quality = ItemQuality.NQ
        if name_expr[-1:].upper() == 'Q':
            name = name_expr[:-2]
            if name_expr[-2:].upper() == 'HQ':
                quality = ItemQuality.HQ
        return Item(name, quality)

@dataclass
class Price():
    nq: int
    hq: int

    def __getitem__(self, quality: ItemQuality):
        if quality is ItemQuality.NQ:
            return self.nq
        if quality is ItemQuality.HQ:
            return self.hq

    def __str__(self) -> str:
        return f'NQ: {self.nq}, HQ: {self.hq}'



@dataclass
class TransactionSummary():
    price: Price
    recent_transaction_nq: int
    recent_transaction_hq: int
    ...

class PriceCacheRec():
    price: Price
    lock: asyncio.Event
    recent_transaction_nq: int
    recent_transaction_hq: int
    world: str

    def __init__(self) -> None:
        self.lock = asyncio.Event()

class PriceCache():
    items: Dict[str, PriceCacheRec]
    def __init__(self) -> None:
        self.items = {}

    async def wait_if_existed(self, name: str):
        if name not in self.items:
            self.items[name] = PriceCacheRec()
            return
        item = self.items[name]
        await item.lock.wait()
        return item

    def set_rec(self, name: str, world: str, price: Price, recent_transaction_nq: int, recent_transaction_hq: int):
        item = self.items[name]
        item.world = world
        item.price = price
        item.recent_transaction_nq = recent_transaction_nq
        item.recent_transaction_hq = recent_transaction_hq
        item.lock.set()

    ...

class RecipeWithCount:
    recipe: 'Recipe'
    count: int

    def __init__(self, *, recipe: 'Recipe', count: int) -> None:
        self.recipe = recipe
        self.count = count

class RequiredCountRepo():
    d: Dict[Item, int]

    def __init__(self) -> None:
        self.d = {}

    def append(self, item: Item, count: int):
        if item in self.d:
            self.d[item] += count
        else:
            self.d[item] = count

class MissingRecipeNode():
    def __init__(self, origin: RecipeWithCount) -> None:
        self.origin = origin
        self.deps: List[MissingRecipeNode] = []
        self.job: Job = None
        self.group = None

    def __str__(self) -> str:
        m = ", ".join([str(m.recipe.item) for m in self.origin.recipe.materials if not m.recipe.selected])
        if m == '':
            m = '<无缺失>'
        star = ''
        if self.origin.recipe.level == 0:
            star = ' ⭐'
        job = self.origin.recipe.jobs[0].job.name
        return f'[{job}] {m} -> {self.origin.recipe.item} x {self.origin.count}{star}'

class MissingRecipeGroup():
    def __init__(self, name: str) -> None:
        self.nodes: List[MissingRecipeNode] = []
        self.deps: Set[MissingRecipeGroup] = set()
        self.name = name
    
    def sort_nodes(self):
        sorted_node = []
        for n in self.nodes:
            self.__sort_nodes(n, sorted_node)
        self.nodes = sorted_node

    def __sort_nodes(self, node: MissingRecipeNode, found: List[MissingRecipeNode]):
        if node in found or node.group is not self:
            return
        for d in node.deps:
            if d not in found:
                self.__sort_nodes(d, found)
        found.append(node)

    def has_loop(self, visited: Set['MissingRecipeGroup'], path: List['MissingRecipeGroup'] = None) -> List['MissingRecipeGroup']:
        if path is None:
            path = []
        if self in path:
            # 找到最后一个self的位置，以那个为起点（包括他）
            print('path', [v.name for v in path])
            print('self', self.name)
            return path[path.index(self):]
        visited.add(self)
        for d in self.deps:
            print(self.name, '->', d.name)
            inner_path = d.has_loop(visited, [*path, self])
            if inner_path is not None:
                return inner_path
        return None

class MaterialRepo():
    d: Dict[Item, RecipeWithCount]
    total_price: int
    total_income: int
    tops: Set['Recipe']
    def __init__(self) -> None:
        self.d = {}
        self.total_price = 0
        self.total_income = 0
        self.tops = set()

    def append(self, item: Item, recipe: 'Recipe', count: int, total_price: int):
        if item in self.d:
            self.d[item].count += count
        else:
            self.d[item] = RecipeWithCount(recipe=recipe, count=count)
        self.total_price += total_price

    def add_income(self, income: int):
        self.total_income += income

    def add_missing_top(self, r: 'Recipe'):
        self.tops.add(r)

    def __resolve_build_step(self, found: List[RecipeWithCount], step: List[RecipeWithCount], r: 'Recipe'):
        cur = None
        li = [f for f in found if f.recipe.item == r.item]
        if len(li) > 0:
            cur = li[0]
        if cur is None:
            print(f'procing {r.item}')
            rwc = RecipeWithCount(recipe=r, count=r.require_count)
            found.append(rwc)
            for ns in [m for m in r.materials if not m.recipe.selected]:
                self.__resolve_build_step(found, step, ns.recipe)
            step.append(rwc)
        else:
            cur.count += r.require_count

    def __sort_group(self, group: MissingRecipeGroup, found: List[MissingRecipeGroup]):
        if group in found:
            return
        for d in group.deps:
            if d not in found:
                self.__sort_group(d, found)
        found.append(group)

    def get_build_step(self):
        found: List[RecipeWithCount] = []
        step: List[RecipeWithCount] = []

        if len(self.tops) > 0:
            for r in self.tops:
                self.__resolve_build_step(found, step, r)
        return step

    def update_group_deps(self, job_grouped_list: List[MissingRecipeGroup]):
        for group in job_grouped_list:
            group.deps = set()
        for group in job_grouped_list:
            for n in group.nodes:
                n.group = group
        for group in job_grouped_list:
            for n in group.nodes:
                for dep in n.deps:
                    if group is not dep.group:
                        group.deps.add(dep.group)
        ...

    def optimize_step(self, step: List[RecipeWithCount]):
        nodes = [MissingRecipeNode(s) for s in step]

        # 处理依赖
        nodes_index_item: Dict[Item, MissingRecipeNode] = {}
        for n in nodes:
            nodes_index_item[n.origin.recipe.item] = n
        
        for n in nodes:
            n.deps = [nodes_index_item[m.recipe.item] for m in n.origin.recipe.materials if not m.recipe.selected]

        # 获取首选职业
        jobs_conut: Dict[Job, int] = {}
        for s in step:
            for j in s.recipe.jobs:
                if j.job in jobs_conut:
                    jobs_conut[j.job] += 1
                else:
                    jobs_conut[j.job] = 1

        for n in nodes:
            print(f'before {n.origin.recipe.jobs}')
            j = sorted([j.job for j in n.origin.recipe.jobs], key=lambda j: jobs_conut[j], reverse=True)[0]
            print(f'j: {j}')
            n.job = j
            n.origin.recipe.jobs = [j for j in n.origin.recipe.jobs if j.job is n.job]
            print(f'set job {n.origin.recipe.item} -> {n.job} {n.origin.recipe.jobs}')

        # 按照职业分组
        job_grouped: Dict[Job, MissingRecipeGroup] = {}
        for n in nodes:
            if n.job not in job_grouped:
                print(f'create group for {n.origin.recipe.item} with job {n.job}')
                job_grouped[n.job] = MissingRecipeGroup(n.job.name)
            job_grouped[n.job].nodes.append(n)
            n.group = job_grouped[n.job]
            print(f'add {n.origin.recipe.item} to group {n.job}')

        print(f'group count {list(job_grouped)}')

        job_grouped_list = list(job_grouped.values())

        # 计算组间依赖
        self.update_group_deps(job_grouped_list)
        # for group in job_grouped_list:
        #     for n in group.nodes:
        #         for dep in n.deps:
        #             if group is not dep.group:
        #                 group.deps.add(dep.group)
        
        # 检查是否有环

        # TODO: 把成环的依赖单独拿出来作为一个组

        # 最多进行10次解环处理

        for i in range(10):
            has_loop = False
            print('contains', [g.name for g in job_grouped_list])

            visited: Set['MissingRecipeGroup'] = set()
            while len(visited) < len(job_grouped_list):
                group = [g for g in job_grouped_list if g not in visited][0]
                print('====组:', group.name, '====')
                endpoints = group.has_loop(visited)
                has_loop = endpoints is not None
                if has_loop:
                    print(f'===存在环!===, 第{i + 1}次')
                    endpoint = endpoints[0]
                    break

            
            if not has_loop:
                break

            print('current', endpoint.name)
            job_grouped_list.remove(endpoint)
            # 把endpoint完全拆开
            # TODO: 识别出造成依赖的所有项，并把他们独立成组合
            for i, n in enumerate(endpoint.nodes):
                g = MissingRecipeGroup(f'{endpoint.name}.{i}')
                g.nodes = [n]
                job_grouped_list.append(g)

            self.update_group_deps(job_grouped_list)
        else:
            return None

        

        # 没有环, 则进行组间排序
        sorted_group: List[MissingRecipeGroup] = []
        for group in job_grouped_list:
            self.__sort_group(group, sorted_group)

        print(f'sorted_group {sorted_group}')
        
        for g in sorted_group:
            g.sort_nodes()

        # sorted_group = list(job_grouped.values())

        result: List[RecipeWithCount] = []
        for g in sorted_group:
            print(f'in group {g.nodes[0].job}')
            for n in g.nodes:
                result.append(n.origin)
                print(f'append result: {n}')
        return result

    def __str__(self) -> str:
        # 按照数量排序，水晶固定放后面
        normals: List[Tuple[Item, RecipeWithCount]] = []
        crystals = []
        for i in self.d.items():
            if i[0].is_crystal():
                crystals.append(i)
            else:
                normals.append(i)
        result = ''
        for op in (normals, crystals):
            op.sort(key=lambda i: i[1].count, reverse=True)
            for i, rc in op:
                from_shop = rc.recipe.single_price == rc.recipe.shop_price
                world_str = f' ({rc.recipe.world})' if rc.recipe.world != '沃仙曦染' and not from_shop else ''
                half = f'{"成品" if len(rc.recipe.materials) > 0 else ""}'
                if len(half) > 0 and rc.recipe.level != 0:
                    half = '半' + half
                if len(half) > 0:
                    half = f' [{half}]'
                result += f'{i} x {rc.count} ~ {round(rc.recipe.single_price)} {"商" if from_shop else ""}{world_str}{half}\n'

        profit_margin = round((self.total_income - self.total_price) / self.total_price * 100)
        
        result += f'\n总成本: {self.total_price}G\n'
        result += f'总收益: {self.total_income}G\n'
        result += f'总利润率: {profit_margin}%\n'

        result += f'\n'
        step = self.get_build_step()

        optimized_step = None
        optimized_step = self.optimize_step(step)

        if optimized_step is not None:
            result += f'(已优化)\n'
            step = optimized_step
            ...

        for s in step:
            m = ", ".join([str(m.recipe.item) for m in s.recipe.materials if not m.recipe.selected])
            if m == '':
                m = '<无缺失>'
            star = ''
            if s.recipe.level == 0:
                star = ' ⭐'
            job = s.recipe.jobs[0].job.name
            result += f'[{job}] {m} -> {s.recipe.item} x {s.count}{star}\n'
            ...

        return result
        
class Recipe():
    connector: aiohttp.TCPConnector = None
    recipe_client: aiohttp.ClientSession = None

    @classmethod
    async def sinit(cls):
        print('init recipt client')
        cls.connector = aiohttp.TCPConnector(limit=20)
        cls.recipe_client = aiohttp.ClientSession(connector=cls.connector, cookies={
            'mogboard_server': f'%E6%B2%83%E4%BB%99%E6%9B%A6%E6%9F%93',
            'mogboard_language': 'chs',
            'mogboard_timezone': f'Asia%2FShanghai',
            'mogboard_homeworld': 'yes'
        })

    def __init__(self, name: str, rule: 'RecipeRule', price_cache: PriceCache, rcRepo: RequiredCountRepo, result_count: int = 1, *, level: int) -> None:
        self.jobs: List[RequiredJob] = []
        self.materials: List['Material'] = []
        self.result_count = result_count
        self.require_count = None
        self.unit_price = None
        self.item = Item.parse(name)
        self.selected = False
        self.shop_price = None
        self.recent_transaction_nq = 0
        self.recent_transaction_hq = 0
        self.price_cache = price_cache
        self.world = None
        self.rule = rule
        self.level = level
        self.rcRepo = rcRepo

    def resolve_require_count(self, count: int = 1):
        times = math.ceil(count / self.result_count) # 制作次数
        self.require_count = times * self.result_count
        self.rcRepo.append(self.item, self.require_count)
        for m in self.materials:
            m.recipe.resolve_require_count(times * m.count)

    async def resolve_unit_price(self):
        origin = self.__collect_unit_price()
        cos = flatten(origin)
        await asyncio.gather(*cos)

    def resolve_shop_price(self):
        item_rec = None
        if self.item.name in db.item_index_singular:
            item_rec = db.item_index_singular[self.item.name][0]
        if item_rec is None:
            raise RuntimeError(f'物品: "{self.item.name}"不存在')
        found_shop_gil = None
        if item_rec.key_as_fk in db.shop_item_gil_index_item:
            found_shop_gil =  db.shop_item_gil_index_item[item_rec.key_as_fk][0]
        if found_shop_gil is not None:
            self.shop_price = found_shop_gil[ShopItemGilKey.Gil]
        for m in self.materials:
            m.recipe.resolve_shop_price()

    def __collect_unit_price(self):
        return [self.__update_unit_price(), [m.recipe.__collect_unit_price() for m in self.materials]]

    @staticmethod
    def norm_detect(data):
        if len(data) == 0:
            return data
        df = pd.DataFrame(data, columns=['value'])
        u = df['value'].mean()
        std = df['value'].std()
        res = kstest(df, 'norm', (u, std))[1]
        if res<=0.05:
            error = df[np.abs(df['value'] - u) > 3 * std]
            data_c = df[np.abs(df['value'] - u) <= 3 * std]
            error_list = error.value.tolist()
            if len(error_list) > 0:
                print(f'异常数据: {error_list}')
            return data_c.value.tolist()
        else:
            return data

    @classmethod
    def strip_bad_val(self, entries: Any):
        vals = [entry['pricePerUnit'] for entry in entries]
        vals_c = self.norm_detect(vals)
        return [entry for entry in entries if entry['pricePerUnit'] in vals_c]
    
    @classmethod
    def strip_bad_val_by(self, entries: Any, field_name: str):
        vals = [entry[field_name] for entry in entries]
        vals_c = self.norm_detect(vals)
        return [entry for entry in entries if entry[field_name] in vals_c]

    @staticmethod
    def classify(entries: Dict[str, Any], by_key: str) -> Dict[str, Dict[str, Any]]:
        classified = {}
        for entry in entries:
            real_key = entry[by_key]
            if real_key not in classified:
                rec = []
                classified[real_key] = rec
            else:
                classified[real_key].append(entry)
        return classified

    @classmethod
    def calc_spec_quality(self, entries: Dict[str, Any], is_hq: str):
        ts_now = time.time()
        ts_half_day_ago = ts_now - 12 * 60 * 60
        ts_7day_ago = ts_now - 7 * 24 * 60 * 60

        xqs = self.strip_bad_val([entry for entry in entries if entry['hq'] == is_hq])
        xqs_c = xqs[:max(5, len([xq for xq in xqs if xq['timestamp'] > ts_half_day_ago]))]
        xq_total_gil = sum([xq['pricePerUnit'] * xq['quantity'] for xq in xqs_c])
        xq_cnt = sum([xq['quantity'] for xq in xqs_c])
        xq_avg = round(xq_total_gil / xq_cnt) if xq_cnt > 0 else None

        recent_transaction_xq = sum([xq['quantity'] for xq in xqs if xq['timestamp'] > ts_7day_ago])
        return xq_avg, recent_transaction_xq

    @classmethod
    def calc_entries_recent_transaction_xq(self, entries: Dict[str, Any], is_hq: str):
        ts_now = time.time()
        ts_7day_ago = ts_now - 7 * 24 * 60 * 60
        xqs = self.strip_bad_val([entry for entry in entries if entry['hq'] == is_hq])
        recent_transaction_xq = sum([xq['quantity'] for xq in xqs if xq['timestamp'] > ts_7day_ago])
        return recent_transaction_xq

    @classmethod
    def calc_transaction(self, entries: Dict[str, Any]) -> TransactionSummary:
        nq_avg, recent_transaction_nq = self.calc_spec_quality(entries, False)
        hq_avg, recent_transaction_hq = self.calc_spec_quality(entries, True)
        price = Price(nq_avg, hq_avg)
        return TransactionSummary(price=price, recent_transaction_nq=recent_transaction_nq, recent_transaction_hq=recent_transaction_hq)

    @classmethod
    def calc_price_xq(self, listings: Dict[str, Any], count: int, is_hq: bool) -> float:
        xqs = self.strip_bad_val_by([listing for listing in listings if listing['hq'] == is_hq], 'pricePerUnit')
        xqs.sort(key=lambda i: i['pricePerUnit'])
        acc_count = 0
        total_price = 0
        for xq in xqs:
            overflow = False
            quantity = xq['quantity']
            if acc_count + quantity >= count:
                quantity = count - acc_count
                overflow = True
            acc_count += quantity
            total_price += xq['pricePerUnit'] * quantity
            if overflow:
                break
        if acc_count == 0: return
        return total_price / acc_count

    @classmethod
    def calc_price(self, listings: Dict[str, Any], count: int) -> float:
        nq = self.calc_price_xq(listings, count, False)
        hq = self.calc_price_xq(listings, count, True)
        return Price(nq, hq)
        ...

    def get_prefer_trans_summary(self, tw: Dict[str, TransactionSummary]) -> Tuple[str, TransactionSummary]:        
        items = list(tw.items())
        items = [item for item in items if item[1].price[self.item.quality] is not None]
        if self.rule.cross_server_query and self.level != 0:
            items = sorted(items, key=lambda x: x[1].price[self.item.quality])
        else:
            items = [item for item in items if item[0] == '沃仙曦染'] + sorted(
                [item for item in items if item[0] != '沃仙曦染'], key=lambda x: x[1].price[self.item.quality]
            )
        return items[0]

    def get_prefer_price_realtime(self, price_of_worlds: Dict[str, Price]):
        items = list(price_of_worlds.items())
        items = [item for item in items if item[1][self.item.quality] is not None]
        if self.rule.cross_server_query and self.level != 0:
            items = sorted(items, key=lambda x: x[1][self.item.quality])
        else:
            items = [item for item in items if item[0] == '沃仙曦染'] + sorted(
                [item for item in items if item[0] != '沃仙曦染'], key=lambda x: x[1][self.item.quality]
            )
        return items[0]

    async def __update_unit_price(self) -> Price:
        cached_result = await self.price_cache.wait_if_existed(self.item.name)
        if cached_result is not None:
            # print('get price of', self.item.name, 'from cache')
            self.unit_price = cached_result.price
            self.world = cached_result.world
            self.recent_transaction_nq = cached_result.recent_transaction_nq
            self.recent_transaction_hq = cached_result.recent_transaction_hq
            return

        item_rec = db.item_index_singular[self.item.name][0]
        # async with aiohttp.ClientSession() as session:
        # print(f'try get price of {self.item}')
        listings = None
        for i in range(5):
            async with self.recipe_client.get(f'https://universalis.app/_next/data/YRQl5w0kO1Rp37hNbxHri/market/{item_rec.key}.json?itemId={item_rec.key}') as response:
                market_resp = await response.json()
                try:
                    listings = market_resp['pageProps']['markets']['陆行鸟']['listings']
                    break
                except:
                    print('market_resp', self.item.name, item_rec.key, market_resp['pageProps']['markets'].keys())
        if listings is None:
            raise RuntimeError(f'获取"{self.item.name}"价格超过最大重试次数')

        
        listings_of_worlds = self.classify(listings, 'worldName')

        price_of_worlds = {world: self.calc_price(listings, self.rcRepo.d[self.item]) for world, listings in listings_of_worlds.items()}
        # world, trans_summary = self.get_prefer_trans_summary(transaction_of_worlds)

        world, price = self.get_prefer_price_realtime(price_of_worlds)

        self.unit_price = price
        self.world = world

        if self.level == 0:
            async with self.recipe_client.get(f'https://universalis.app/api/history/%E9%99%86%E8%A1%8C%E9%B8%9F/{item_rec.key}?entries=100') as response:
                resp = await response.json()

            try:
                entries_of_woxian = self.classify(resp['entries'], 'worldName')['沃仙曦染']
                self.recent_transaction_nq = self.calc_entries_recent_transaction_xq(entries_of_woxian, False)
                self.recent_transaction_hq = self.calc_entries_recent_transaction_xq(entries_of_woxian, True)
            except:
                ...
        self.price_cache.set_rec(self.item.name, self.world, self.unit_price, self.recent_transaction_nq, self.recent_transaction_hq)
        print(f'{self.item}', self.unit_price)

    def check_min_cost_node(self):
        if self.min_cost_of() == self.total_price:
            self.selected = True
        else:
            for m in self.materials:
                m.recipe.check_min_cost_node()

    def min_cost_of(self):
        if len(self.materials) > 0:
            return min(self.total_price, sum([m.recipe.min_cost_of() for m in self.materials]))
        else:
            return self.total_price

    @property
    def single_price(self):
        price = self.unit_price[self.item.quality]
        if self.item.quality is ItemQuality.NQ and self.shop_price is not None:
            if price is None:
                price = self.shop_price
            else:
                price = min(self.shop_price, price)
        if price is None:
            raise RuntimeError(f'未查询到物品"{self.item}"的价格')
        return price
        ...

    @property
    def total_price(self):
        return self.single_price * self.require_count

    def gather_material(self, repo: MaterialRepo):
        if self.selected:
            repo.append(self.item, self, self.require_count, self.total_price)
        if not self.selected and self.level == 0:
            repo.add_missing_top(self)
        for m in self.materials:
            m.recipe.gather_material(repo)

    async def parse(self, *decos: Callable[[str], Awaitable[Any]]) -> List[str]: # 需要的成品个数
        s = []

        obtain_ways = []
        if len(self.jobs) > 0:
            obtain_ways = self.jobs
        else:
            item = db.get_gathering_item(self.item.name)
            if item is not None:
                obtain_ways = [item.job]
        if self.shop_price is not None:
            obtain_ways.append(f'商店:{self.shop_price}G')
        if len(obtain_ways) == 0:
            obtain_ways.append('未知')
        world_str = f'({self.world})' if self.world != '沃仙曦染' else ''
        formatted_item = f'[{"√" if self.selected else "x"}]{self.item} x {self.require_count} [{" ".join([str(j) for j in obtain_ways])}]{" ".join([""] + [str(r) for r in await asyncio.gather(*[d(self.item.name) for d in decos])])} {world_str}'
        if self.unit_price is not None:
            formatted_item += f' -> {round(self.total_price)}G'
        s.append(formatted_item)

        sub = []
        for m in self.materials:
            sub.extend(['==' + f'{r}' for r in await m.recipe.parse(*decos)])
        s.extend(sub)
        return s

    @classmethod
    def build(self, name: str, rule: 'RecipeRule', price_cache: PriceCache, rcRepo: RequiredCountRepo, result_count: int = 1, *, level: int = 0):
        r = Recipe(name, rule, price_cache, rcRepo, result_count, level=level)
        matched = False
        if r.item.name in db.recipe_index_result:
            for recipe in db.recipe_index_result[r.item.name]:
                job = recipe[RecipeKey.CraftType]
                rj = RequiredJob()
                rj.job = Job[job]
                rj.level = recipe[RecipeKey.RecipeLevelTable].query()[RecipeLevelTableKey.ClassJobLevel]
                r.jobs.append(rj)
                if not matched:
                    matched = True
                    r.result_count = max(recipe[RecipeKey.AmountResult], r.result_count)
                    for i in range(10): #8
                        item_name = recipe[f'Item{{Ingredient}}[{i}]']
                        item_amount = recipe[f'Amount{{Ingredient}}[{i}]']
                        if item_amount > 0:
                            r.materials.append(Material(self.build(item_name, rule, price_cache, rcRepo, level=level+1), item_amount * r.result_count))
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

class RecipeRule():
    def __init__(self) -> None:
        self.force_use_items: List[Item] = []
        self.cross_server_query = False
        self.only_summary = False

    class State(Enum):
        ForceUse = '使用' # 指定最终配方强制使用某种材料
        CrossServer = '跨服' # 启用后将计算全区平均价格
        OnlySummary = '概览'
        Unknown = auto()
        ...

    @classmethod
    def of(self, *lex: str):
        rule = RecipeRule()
        state = self.State.Unknown
        state_strs = [v.value for v in self.State if type(v.value) is str]
        for l in lex:
            if l in state_strs:
                state = self.State(l)
            if state is self.State.Unknown:
                raise RuntimeError(f'请指定规则名, 可选: {",".join(state_strs)}')
            if state is self.State.ForceUse:
                rule.force_use_items.append(Item.parse(l))
                continue
            if state is self.State.CrossServer:
                rule.cross_server_query = True
                state = self.State.Unknown
                continue
            if state is self.State.OnlySummary:
                rule.only_summary = True
                state = self.State.Unknown
                continue
        return rule

@dataclass
class Say():
    text: str
    comefrom: str
    ...

def rnd_str(len: int = 5):
    return "".join(random.sample("zyxwvutsrqponmlkjihgfedcba0123456789",len))

def to_vp_params(param_str: str) -> Dict[str, int]:
    param_str = param_str.replace(' ', '')
    params = param_str.split(',')
    params = [p.split('=')[:2] for p in params]
    params = [p if p[0] not in ('width', 'height') else [p[0], int(p[1])] for p in params]
    return dict(params)

class MacroEngine():
    def __init__(self) -> None:
        self.vars = {
            '580HQ_服装_T职': '古典风格剑斗士头甲HQ,古典风格剑斗士兜甲HQ,古典风格剑斗士袖甲HQ,古典风格剑斗士三角裤HQ,古典风格剑斗士胫甲HQ',
            '580HQ_首饰_T职': '古典风格御敌耳坠HQ,古典风格御敌项环HQ,古典风格御敌手环HQ,古典风格御敌指环HQ*2',
            '580HQ_十件套_T职': '$580HQ_服装_T职,$580HQ_首饰_T职',
            '580HQ_主副手_战士': '古典风格战斧HQ',
            '580HQ_主副手_骑士': '古典风格长剑HQ,古典风格长盾HQ',
            '580HQ_主副手_暗黑骑士': '古典风格巨剑HQ',
            '580HQ_主副手_绝枪战士': '古典风格枪刃HQ',

            '580HQ_战士': '$580HQ_十件套_T职,$580HQ_主副手_战士',
            '580HQ_骑士': '$580HQ_十件套_T职,$580HQ_主副手_骑士',
            '580HQ_暗黑骑士': '$580HQ_十件套_T职,$580HQ_主副手_暗黑骑士',
            '580HQ_绝枪战士': '$580HQ_十件套_T职,$580HQ_主副手_绝枪战士',
            '580HQ_T职': '$580HQ_十件套_T职,$580HQ_主副手_战士,$580HQ_主副手_骑士,$580HQ_主副手_暗黑骑士,$580HQ_主副手_绝枪战士',

            '580HQ_服装_H职': '古典风格医护兵月桂冠HQ,古典风格医护兵长衣HQ,古典风格医护兵腕环HQ,古典风格医护兵三角裤HQ,古典风格医护兵战靴HQ',
            '580HQ_首饰_H职': '古典风格治愈耳坠HQ,古典风格治愈项环HQ,古典风格治愈手环HQ,古典风格治愈指环HQ*2',
            '580HQ_十件套_H职': '$580HQ_服装_H职,$580HQ_首饰_H职',
            '580HQ_主副手_白魔': '古典风格牧杖HQ',
            '580HQ_主副手_占星': '古典风格黄道仪HQ',
            '580HQ_主副手_学者': '古典风格魔导典HQ',
            '580HQ_主副手_贤者': '古典风格蛇石针HQ',
            
            '580HQ_白魔': '$580HQ_十件套_H职,$580HQ_主副手_白魔',
            '580HQ_占星': '$580HQ_十件套_H职,$580HQ_主副手_占星',
            '580HQ_学者': '$580HQ_十件套_H职,$580HQ_主副手_学者',
            '580HQ_贤者': '$580HQ_十件套_H职,$580HQ_主副手_贤者',
            '580HQ_H职': '$580HQ_十件套_H职,$580HQ_主副手_白魔,$580HQ_主副手_占星,$580HQ_主副手_学者,$580HQ_主副手_贤者',

            '580HQ_服装_远敏': '古典风格弓斗士头甲HQ,古典风格弓斗士长衣HQ,古典风格弓斗士腕环HQ,古典风格弓斗士三角裤HQ,古典风格弓斗士战靴HQ',
            '580HQ_首饰_远敏': '古典风格精准耳坠HQ,古典风格精准项环HQ,古典风格精准手环HQ,古典风格精准指环HQ*2',
            '580HQ_十件套_远敏': '$580HQ_服装_远敏,$580HQ_首饰_远敏',
            '580HQ_主副手_诗人': '古典风格骑兵弓HQ',
            '580HQ_主副手_机工': '古典风格手炮HQ',
            '580HQ_主副手_舞者': '古典风格圆月轮HQ',

            '580HQ_诗人': '$580HQ_十件套_远敏,$580HQ_主副手_诗人',
            '580HQ_机工': '$580HQ_十件套_远敏,$580HQ_主副手_机工',
            '580HQ_舞者': '$580HQ_十件套_远敏,$580HQ_主副手_舞者',
            '580HQ_远敏': '$580HQ_十件套_远敏,$580HQ_主副手_诗人,$580HQ_主副手_机工,$580HQ_主副手_舞者',

            '580HQ_服装_法系': '古典风格旗手角冠HQ,古典风格旗手长衣HQ,古典风格旗手半指护手HQ,古典风格旗手裙裤HQ,古典风格旗手战靴HQ',
            '580HQ_首饰_法系': '古典风格咏咒耳坠HQ,古典风格咏咒项环HQ,古典风格咏咒手环HQ,古典风格咏咒指环HQ*2',
            '580HQ_十件套_法系': '$580HQ_服装_法系,$580HQ_首饰_法系',
            '580HQ_主副手_黑魔': '古典风格长玉杖HQ',
            '580HQ_主副手_赤魔': '古典风格小剑HQ',
            '580HQ_主副手_召唤': '古典风格魔导书HQ',

            '580HQ_黑魔': '$580HQ_十件套_法系,$580HQ_主副手_黑魔',
            '580HQ_赤魔': '$580HQ_十件套_法系,$580HQ_主副手_赤魔',
            '580HQ_召唤': '$580HQ_十件套_法系,$580HQ_主副手_召唤',
            '580HQ_法系': '$580HQ_十件套_法系,$580HQ_主副手_黑魔,$580HQ_主副手_赤魔,$580HQ_主副手_召唤',

            # 武僧 武士
            '580HQ_服装_拳斗士': '古典风格拳斗士面具HQ,古典风格拳斗士兜甲HQ,古典风格拳斗士袖甲HQ,古典风格拳斗士三角裤HQ,古典风格拳斗士战靴HQ',
            # 龙骑士 钐镰客
            '580HQ_服装_骑士': '古典风格骑士头甲HQ,古典风格骑士长衣HQ,古典风格骑士袖甲HQ,古典风格骑士三角裤HQ,古典风格骑士战靴HQ',
            # 忍者
            '580HQ_服装_忍者': '古典风格双剑斗士面具HQ,古典风格双剑斗士兜甲HQ,古典风格双剑斗士袖甲HQ,古典风格双剑斗士三角裤HQ,古典风格双剑斗士战靴HQ',
            # 武僧 龙骑士 武士 钐镰客
            '580HQ_首饰_强攻': '古典风格强攻耳坠HQ,古典风格强攻项环HQ,古典风格强攻手环HQ,古典风格强攻指环HQ*2',

            '580HQ_十件套_拳斗士': '$580HQ_服装_拳斗士,$580HQ_首饰_强攻',
            '580HQ_十件套_骑士': '$580HQ_服装_骑士,$580HQ_首饰_强攻',
            '580HQ_十件套_忍者': '$580HQ_服装_忍者,$580HQ_首饰_远敏',

            '580HQ_主副手_武僧': '古典风格旋棍HQ',
            '580HQ_主副手_武士': '古典风格武士刀HQ',
            '580HQ_主副手_龙骑': '古典风格长枪HQ',
            '580HQ_主副手_镰刀': '古典风格战镰HQ',
            '580HQ_主副手_忍者': '古典风格小刀HQ',

            '580HQ_武僧': '$580HQ_十件套_拳斗士,$580HQ_主副手_武僧',
            '580HQ_武士': '$580HQ_十件套_拳斗士,$580HQ_主副手_武士',
            '580HQ_龙骑': '$580HQ_十件套_骑士,$580HQ_主副手_龙骑',
            '580HQ_镰刀': '$580HQ_十件套_骑士,$580HQ_主副手_镰刀',
            '580HQ_忍者': '$580HQ_十件套_忍者,$580HQ_主副手_忍者',
        }

    def parse(self, text: str):
        expr = '\\$([\u4e00-\u9fa5_a-zA-Z0-9]+)'
        proc = text
        def rep(m):
            return self.vars[m.group(1)]
        while re.match(expr, proc) is not None:
            proc = re.sub(expr, rep, proc)
        return proc

class ItemCountExpr():
    item_name: str
    count: int
    def __init__(self, expr: str) -> None:
        parsed = expr.split('*')
        self.item_name = parsed[0]
        if len(parsed) > 1:
            self.count = int(parsed[1])
        else:
            self.count = 1
        pass
    
    ...

class FF(Plugin):
    def __init__(self) -> None:
        self.macro_engine = MacroEngine()
        super().__init__('ff')

    def get_resolvers(self):
        return {
            RecipeRule: RecipeRule.of
        }

    @autorun
    async def recipe_init(self):
        await Recipe.sinit()

    async def mrmy(self):
        async with aiohttp.ClientSession() as session:
            result = Say('咱们尼格有力量!', '佚名')
            url = f'https://eolink.o.apispace.com/mingrenmingyan/api/v1/ming_ren_ming_yan/random?page_size=1'
            async with session.get(url, headers={
                'X-APISpace-Token': 'n1bk048xsp53crpwfjlh4pq8267fvv7d',
                'Authorization-Type': 'apikey',
            }) as response:
                resp = await response.json()
        if resp['code'] == 200:
            data = resp['data'][0]
            result = Say(data['text'], data["comefrom"])
        return result

    @staticmethod
    async def query_pic(say: Say, file = None):
        if file is None:
            file = os.path.join(TMP_PATH, f'{rnd_str()}.png')
        browser = await launch()
        page = await browser.newPage()
        pic_id = random.randint(0, 8)
        await page.goto(f'http://127.0.0.1:5000/ad/{pic_id}?text={urllib.parse.quote_plus(say.text)}&comefrom={urllib.parse.quote_plus(say.comefrom)}')

        # viewport = await page.querySelector('meta[name="viewport"]')
        # viewport_content = await viewport.getProperty('content')
        # viewport_content = await viewport_content.jsonValue()
        # vp_params = to_vp_params(viewport_content)
        # await page.setViewport({
        #     'width': vp_params['width'],
        #     'height': vp_params['height'],
        # })
        await page.setViewport({
            'width': 1000,
            'height': 600,
        })

        await page.screenshot({
            'path': file,
            'omitBackground': True,
        })

        await browser.close()
        return file

    @instr('师说')
    async def adl_say(self):
        say = await self.mrmy()
        while len(say.text) > 58:
            say = await self.mrmy()
        file = await self.query_pic(say)
        return [
            mirai.models.message.Image(path=file)
        ]

    def recipe_tree_preprocess(self, item: ItemCountExpr, rule: RecipeRule, price_cache: PriceCache, rcRepo: RequiredCountRepo):
        recipe_tree = Recipe.build(item.item_name, rule, price_cache, rcRepo, item.count)
        recipe_tree.resolve_require_count()
        return recipe_tree

    async def get_recipe_rec(self, recipe_tree: Recipe, repo: MaterialRepo):
        recipe_tree.resolve_shop_price()
        await recipe_tree.resolve_unit_price()
        recipe_tree.check_min_cost_node()

        cost = recipe_tree.min_cost_of()
        recipe_tree.gather_material(repo)
        repo.add_income(recipe_tree.total_price)
        
        price = recipe_tree.total_price
        profit_margin = round((price - cost) / cost * 100)
        result = ''
        result += '\n'.join(await recipe_tree.parse())
        result += '\n\n'
        result += f'成本估计: {cost}G\n价格估计: {price}G\n利润率: {profit_margin}%\n'
        result += f'7日内售出 HQ:{recipe_tree.recent_transaction_hq} NQ:{recipe_tree.recent_transaction_nq}'
        return result

    @instr('配方')
    async def recipe(self, event: MessageEvent, name: str, expr: RecipeRule):        
        print(f'配方 {event.sender.get_name()}({"群聊" if type(event) is GroupMessage else "私聊"}) -> {name}')
        price_cache = PriceCache()
        repo = MaterialRepo()
        rcRepo = RequiredCountRepo()

        prev = datetime.datetime.now()

        trees = [self.recipe_tree_preprocess(ItemCountExpr(n), expr, price_cache, rcRepo) for n in self.macro_engine.parse(name).split(',')]
        results = await asyncio.gather(*[self.get_recipe_rec(t, repo) for t in trees])
        
        span = datetime.datetime.now() - prev
        
        print('done.')

        say = await self.mrmy()
        while len(say.text) > 58:
            say = await self.mrmy()
        file = await self.query_pic(say)
        return [
            Forward(node_list=[
                ForwardMessageNode.create(
                    event.sender, 
                    MessageChain([str(repo)])
                ),
                *[
                    ForwardMessageNode.create(
                        event.sender, 
                        MessageChain([result])
                    ) for result in (results if not expr.only_summary else [])
                ],
                ForwardMessageNode.create(
                    event.sender, 
                    MessageChain([f'耗时: {span.total_seconds():.2f}秒'])
                ),
                ForwardMessageNode.create(
                    Friend(id=1293103235, nickname='阿德勒'),
                    MessageChain([
                        mirai.models.message.Image(path=file)
                    ])
                )
            ])
        ]
        # return [result]
        
    @instr('价格')
    async def price(self, event: MessageEvent, name: str, span: int = 1):
        if type(event) is GroupMessage:
            raise RuntimeError('请私聊机器人使用本命令')

        # 查询n日内各服的平均价格
        item_rec = db.item_index_singular[name][0]
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://universalis.app/api/history/%E9%99%86%E8%A1%8C%E9%B8%9F/{item_rec.key}?entries=1000') as response:
                resp = await response.json()
        entries = resp['entries']
        entry_of_worlds = {}
        ts_now = time.time()
        ts_nday_ago = ts_now - span * 24 * 60 * 60

        # 分类
        for entry in entries:
            world_name = entry['worldName']
            if world_name not in entry_of_worlds:
                rec = []
                entry_of_worlds[world_name] = rec
            else:
                rec = entry_of_worlds[world_name]
            if entry['timestamp'] > ts_nday_ago:
                # print(entry)
                rec.append(entry)

        info_of_world = {}
        for world_name, entries in entry_of_worlds.items():
            nqs = [entry for entry in entries if not entry['hq']]
            hqs = [entry for entry in entries if entry['hq']]
            nq_total_gil = sum([nq['pricePerUnit'] * nq['quantity'] for nq in nqs])
            hq_total_gil = sum([hq['pricePerUnit'] * hq['quantity'] for hq in hqs])
            nq_cnt = sum([nq['quantity'] for nq in nqs])
            hq_cnt = sum([hq['quantity'] for hq in hqs])
            info_of_world[world_name] = {
                'nq_avg': round(nq_total_gil / nq_cnt) if nq_cnt > 0 else '无',
                'nq_cnt': nq_cnt,
                'hq_avg': round(hq_total_gil / hq_cnt) if hq_cnt > 0 else '无',
                'hq_cnt': hq_cnt,
            }
        
        res = []
        res.append(f'==={name}===')
        for world_name, info in info_of_world.items():
            res.append(f'{world_name} NQ({info["nq_cnt"]}): {info["nq_avg"]}, HQ({info["hq_cnt"]}): {info["hq_avg"]}')
        
        return ['\n'.join(res)]


    @instr('合并')
    async def comb(self, event: MessageEvent):
        return [
            Forward(node_list=[
                ForwardMessageNode.create(
                    event.sender, 
                    MessageChain(['hi'])
                )
            ])
        ]

    class MacroSubCmd(Enum):
        Var = '变量'
        Exec = '执行'
        ...

    @instr('宏')
    async def var(self, sub_cmd: MacroSubCmd, term: str=None):
        if sub_cmd is self.MacroSubCmd.Var:
            return ', '.join( self.macro_engine.vars.keys())
        if sub_cmd is self.MacroSubCmd.Exec:
            return self.macro_engine.parse(term)
        

    class SalarySubCmd(Enum):
        Start = '开始'
        Stop = '结束'
        ...

    @instr('打工')
    async def salary(self, sub_cmd: SalarySubCmd, img: Image):
        path = await img.download(os.path.join(TMP_PATH, f'{rnd_str()}.jpg'))
        # result = reader.readtext(str(path), allowlist='0123456789,', decoder='wordbeamsearch', contrast_ths=0.8)
        # print(result)
        # if sub_cmd is self.SalarySubCmd.Start:
        #     ...
        # if sub_cmd is self.SalarySubCmd.Stop:
        #     ...