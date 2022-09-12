from mirai import FriendMessage, GroupMessage, MessageEvent
from ..plugin import Plugin, autorun, instr

class CustomArg():
    def __init__(self, x) -> None:
        self.x = x
    ...

    def __str__(self) -> str:
        return f'CA {self.x}'

class Test(Plugin):
    def __init__(self) -> None:
        super().__init__('测试')

    def get_resolvers(self):
        def resolve(x: int = 6):
            return CustomArg(x)
        return {
            CustomArg: resolve
        }

    @instr('默认')
    async def create(self, event: FriendMessage, dd: CustomArg):
        return [
            f'值: {dd}'
        ]