import asyncio
import json
import urllib
import urllib.parse
import telebot
import os
import requests

telegram_bot_token = os.getenv("telegram_bot_api")
bot = telebot.TeleBot(telegram_bot_token)

@bot.message_handler(commands=["start"]) #带斜杠的指令”\start"
def statr_message(message):
    #bot.reply_to(message,"你好") #引用回复
    bot.send_message(message.chat.id,"你好，老朽是陳大師!")

@bot.message_handler(func=lambda message:True)
def echo_call(message):
    try:
        encoded_text = urllib.parse.quote(message.text)
        response = requests.post(
            f"http://localhost:8000/chat",
            params={
                "query": encoded_text,
                "chat_id": str(message.chat.id)  # 关键：传递 chat.id
            },
            timeout=500
        )
        if response.status_code == 200:
            AI_message = json.loads(response.text)
            bot.send_message(message.chat.id,AI_message["res"]["output"].encode("utf-8"))
            voice_path = f"{AI_message['id']}.mp3"
            asyncio.run(check_voice_path(message,audio_path=voice_path))
    except requests.RequestException as e:
        bot.send_message(message.chat.id, "请求超时啦!")

async def check_voice_path(message,audio_path): #异步循环检查
    while True:
        if os.path.exists(audio_path):
            with open(audio_path,"rb") as f:
                bot.send_audio(message.chat.id,f)
            os.remove(audio_path) #删除文件
            break
        else:
            print("waiting voice")
            await asyncio.sleep(1) #停顿一秒钟

bot.infinity_polling() #定时器，无限循环检查动作