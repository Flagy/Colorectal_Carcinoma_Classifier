import time
import telepot
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineQueryResultArticle, InputTextMessageContent, InlineQueryResultPhoto
from Token import TOKEN
import os

def on_inline_query(msg):
    def compute():
        query_id, from_id, query_string = telepot.glance(msg, flavor='inline_query')
        print('Inline Query:', query_id, from_id, query_string)

        articles = [InlineQueryResultArticle(
                        id='abc',
                        title='ciao',
                        input_message_content=InputTextMessageContent(message_text='suca')),
                    InlineQueryResultPhoto(
                        type = 'photo',
                        id = 'pic',
                        photo_url = 'C:/Users/super/OneDrive/Documenti/Project_bioinfo/images/training/H/image_H_1.jpg'

                        )]

        return articles

    answerer.answer(msg, compute)

def on_chosen_inline_result(msg):
    result_id, from_id, query_string = telepot.glance(msg, flavor='chosen_inline_result')
    print ('Chosen Inline Result:', result_id, from_id, query_string)
    

bot = telepot.Bot(TOKEN)
answerer = telepot.helper.Answerer(bot)

MessageLoop(bot, {'inline_query': on_inline_query,
                  'chosen_inline_result': on_chosen_inline_result}).run_forever()

while 1:
    time.sleep(10)
