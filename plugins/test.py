from mirai import FriendMessage, GroupMessage, MessageChain, MessageEvent
from ..plugin import Plugin, autorun, instr
from mirai.models.message import Forward, ForwardMessageNode
from mirai.models.entities import Friend
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

    @instr('假消息')
    async def create(self, id: str, name: str, msg: str):
        return [
            Forward(node_list=[
                ForwardMessageNode.create(
                    Friend(id=id, nickname=name), 
                    MessageChain([msg])
                )
            ])
        ]