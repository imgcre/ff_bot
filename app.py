from typing import Dict
from activator import AtActivator
from mirai import At, GroupMessage, Mirai, MessageEvent, Plain, WebSocketAdapter
import plugin
import inspect

bot = Mirai(3579656756, adapter=WebSocketAdapter(
    verify_key='INITKEYzMKXIKXq', host='localhost', port=8080
))

activator = AtActivator(bot.qq)
# 激活方式

engine = plugin.Engine()

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
        await send(
            f' 错误{ctx.pretty_stack()}: ',
            *e.args
        )

def main():
    engine.load(bot)
    bot.run()

if __name__ == '__main__':
    main()