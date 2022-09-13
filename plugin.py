import importlib.util
import inspect
from typing import Any, Callable, Final, Dict, List, Tuple, Union, get_args, get_origin

import glob

from mirai import MessageChain, MessageEvent, Mirai, Plain

PLUGIN_PATH: Final[str] = './plugins/*.py'

class Plugin():
    __BOT_PLUGIN__ = True
    bot: Mirai
    def __init__(self, cmd) -> None:
        self.cmd = cmd

    def init(self, bot: Mirai):
        self.bot = bot
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_bot_autorun_'):
                bot.add_background_task(method)

    def get_resolvers(self) -> Dict[type, Callable[[Any], Any]]: return {}

def is_plugin(cls):
    try:
        return cls.__BOT_PLUGIN__ and cls.__name__ != Plugin.__name__
    except:
        return False

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

class Engine():
    plugins: Dict[str, Plugin] = {}
    def load(self, bot: Mirai):
        for file in glob.glob(PLUGIN_PATH):
            mod_name = file.replace('\\', '/').replace('./', '.').replace('/', '.')[:-3]
            spec = importlib.util.spec_from_file_location(mod_name, file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for _, member in inspect.getmembers(mod, inspect.isclass):
                if is_plugin(member):
                    p: Plugin = member()
                    p.init(bot)
                    self.plugins[p.cmd] = p

    def context(self):
        return Context(self)
    
class ResolveFailedException(Exception):
    ...

class Context():
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.stack = []
        self.instr_attrs = []

    async def exec(self, event: MessageEvent, chain: MessageChain):
        processed_chain = self.preprocess(chain)
        try:
            plugin_name = self.get_text(processed_chain[0])
        except:
            raise RuntimeError('请指定插件名')
        try:
            plugin = self.engine.plugins[plugin_name]
        except:
            raise RuntimeError(f'插件"{plugin_name}"不存在')

        self.stack.append(plugin_name)

        instr_name = None
        if len(processed_chain) > 0:
            try:
                instr_name = self.get_text(processed_chain[1])
            except:
                raise RuntimeError('子命令无效')

        for _, method in inspect.getmembers(plugin, predicate=inspect.ismethod):
            if hasattr(method, '_instr_name_') and method._instr_name_ == instr_name:
                self.stack.append(instr_name)
                self.instr_attrs = method._instr_attrs_
                s = inspect.signature(method)
                params = [p for p in s.parameters.values()]
                if not self.is_target_msg(params, event):
                    raise RuntimeError(f'无法在当前上下文中调用')
                return await method(*self.resolve_args(method, processed_chain[2:], event, plugin))

        raise RuntimeError(f'不存在子命令"{instr_name}"')

    def pretty_stack(self):
        return ''.join([f'[{i}]' for i in self.stack])

    @staticmethod
    def is_optional(t):
        origin = get_origin(t)
        args = get_args(t)
        return origin is Union and len(args) == 2 and args[1] is type(None)

    def resolve_args(self, method, chain, event: MessageEvent, plugin: Plugin):
        s = inspect.signature(method)
        params = [p for p in s.parameters.values()]
        if not self.is_target_msg(params, event):
            raise RuntimeError(f'无法在当前上下文中调用')
        args = []
        resolvers = plugin.get_resolvers()
        for p in params:
            if self.is_type_of(p.annotation, MessageEvent):
                args.append(event)
            elif self.is_type_of(p.annotation, '.plugin.Context'):
                args.append(self)
            else:
                def append_single_arg():
                    anno = p.annotation
                    will_skip = False
                    if self.is_optional(anno):
                        will_skip = True
                        anno = get_args(anno)[0]
                    if  p.default is not None:
                        will_skip = True
                    try:
                        if anno in resolvers:
                            resolver = resolvers[anno]
                            sub_args = self.resolve_args(resolver, chain, event, plugin)
                            try:
                                sub_res = resolver(*sub_args)
                            except Exception as e:
                                raise ResolveFailedException(e)
                            args.append(sub_res)
                            return
                        if len(chain) == 0:
                            raise RuntimeError(f'参数不足')
                        front = chain.pop(0)
                        if anno in (str, int, float):
                            if type(front) is not str or isinstance(front, Plain):
                                raise RuntimeError(f'参数类型错误')
                            args.append(anno(front.text if isinstance(front, Plain) else front))
                            return
                        raise RuntimeError(f'无法识别的参数类型')
                    except ResolveFailedException as e:
                        raise e.args[0] from e
                    except Exception as e:
                        if not will_skip: raise
                        default = p.default
                        if default is inspect._empty:
                            default = None
                        args.append(default)
                if p.kind is inspect._ParameterKind.VAR_POSITIONAL:
                    while len(chain) > 0:
                        append_single_arg()
                else:
                    append_single_arg()
        return args

    @staticmethod
    def get_allowed_events(param: inspect.Parameter) -> Tuple:
        if get_origin(param.annotation) is Union:
            union_args = get_args(param.annotation)
            all_msg_type = all([Context.is_type_of(a, MessageEvent) for a in union_args])
            all_not_msg_type = all([not Context.is_type_of(a, MessageEvent) for a in union_args])
            assert(all_msg_type or all_not_msg_type)
            if all_msg_type:
                return union_args
        elif Context.is_type_of(param.annotation, MessageEvent):
            return (param.annotation,)

    @staticmethod
    def is_type_of(var, cls):
        if type(cls) is str:
            return inspect.isclass(var) and '.'.join([var.__module__, var.__qualname__]) == cls
        return inspect.isclass(var) and issubclass(var, cls)

    @staticmethod
    def is_target_msg(params: List[inspect.Parameter], event: MessageEvent):
        for p in params:
            allowed_events = Context.get_allowed_events(p)
            if allowed_events is None: continue
            return any([isinstance(event, e) for e in allowed_events])
        return True

    @staticmethod
    def get_text(comp):
        if isinstance(comp, str):
            return comp
        assert(isinstance(comp, Plain))
        return comp.text

    @staticmethod
    def preprocess(chain: MessageChain):
        res = []
        for msg in chain:
            if isinstance(msg, Plain):
                res.extend(list(filter(lambda x: x != '', msg.text.split(' '))))
            else:
                res.append(msg)
        return res


def instr(name, *attr):
    def wrapper(func):
        func._instr_name_ = name
        func._instr_attrs_ = flatten(list(attr))
        return func
    return wrapper

def autorun(func):
    func._bot_autorun_ = True
    return func
