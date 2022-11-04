import asyncio
import os
from mirai import FriendMessage, GroupMessage, MessageChain, MessageEvent
from ..plugin import Plugin, autorun, instr
from mirai.models.message import Forward, ForwardMessageNode, Image
from mirai.models.entities import Friend

RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'you')

class Test(Plugin):
    def __init__(self) -> None:
        super().__init__('你')

    @instr('还活着吗')
    async def create(self):
        return [
            Image(path=os.path.join(RESOURCE_PATH, '0.jpg'))
        ]