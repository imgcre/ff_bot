import traceback
from typing import Dict
from activator import AtActivator
from mirai import At, GroupMessage, Mirai, MessageEvent, Plain, WebSocketAdapter
import plugin
import inspect
from mirai.models.events import BotInvitedJoinGroupRequestEvent, GroupRecallEvent
from mirai.models.api import RespOperate

bot = Mirai(3579656756, adapter=WebSocketAdapter(
    verify_key='INITKEYzMKXIKXq', host='localhost', port=8080
))

activator = AtActivator(bot.qq)
# 激活方式

engine = plugin.Engine()

@bot.on(GroupRecallEvent)
async def on_recall(event: GroupRecallEvent):
    if event.group.id != 563158420:
        return
    resp = await bot.message_from_id(event.message_id)
    msg: GroupMessage = resp.data
    await bot.send(event.group, [f'{msg.sender.get_name()}({msg.sender.id})撤回了一条消息:', *msg.message_chain])

@bot.on(MessageEvent)
async def on_message(event: MessageEvent):
    chain = activator.check(event)
    if chain is None: return

    ctx = engine.context()
    
    async def send(*args):
        res = []
        if isinstance(event, GroupMessage):
            res.append(At(target=event.sender.id))
            res.append('\n')
        res.extend(args)
        await bot.send(event, res)

    try:
        res = await ctx.exec(event, chain)
        if res is not None:
            await send(*res)
    except Exception as e:
        # raise
        traceback.print_exc()
        await send(
            f' 错误{ctx.pretty_stack()}: ',
            *e.args
        )

def main():
    engine.load(bot)
    bot.run()

if __name__ == '__main__':
    main()