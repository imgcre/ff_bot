from abc import ABC, abstractmethod
from mirai import At, FriendMessage, MessageEvent, MessageChain, GroupMessage

class Activator(ABC):
    @abstractmethod
    def check(self, event: MessageEvent) -> MessageChain: ...

class AtActivator(Activator):
    def __init__(self, bot_id: int) -> None:
        self.bot_id = bot_id

    def check(self, event: MessageEvent) -> MessageChain: 
        if isinstance(event, GroupMessage):
            if len(event.message_chain) > 1 and event.message_chain[1] == At(self.bot_id):
                return event.message_chain[2:]
            else: 
                return None
        elif isinstance(event, FriendMessage):
            return event.message_chain[1:]
        return None
