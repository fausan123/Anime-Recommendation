from telegram import ReplyKeyboardMarkup, Update, ReplyKeyboardRemove, ForceReply
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

from decouple import config
import pandas as pd
import numpy as np
import pickle
import os

PORT = int(os.environ.get('PORT', 8443))
BOT_TOKEN = os.environ.get('BOT_TOKEN')
animelist = pd.read_csv('data/anime.csv')
knn_file = open("data/knnmodel", "rb")
knnmodel = pickle.load(knn_file)
sparsey_file = open("data/sparsey", "rb")
sparse_y = pickle.load(sparsey_file)
animeids_file = open("data/animeids", "rb")
anime_ids = pickle.load(animeids_file)

knn_file.close()
sparsey_file.close()
animeids_file.close()


def get_animenames(ids: list):
  info = []
  for id in ids:
    details = animelist[animelist["MAL_ID"] == id]
    name = details.at[details.index[0], 'English name']
    if (name == "Unknown"):
        name = details.at[details.index[0], 'Name']
    genre = details.at[details.index[0], 'Genres']
    info.append((name, genre))
  return info

def get_knnrecommendation(knn_model, anime_id, y):
  distances, indices = knn_model.kneighbors(y[anime_ids[anime_id], :].reshape(1, -1), n_neighbors=11)
  reverse_ids = {v: k for k, v in anime_ids.items()}
  anime_list = [reverse_ids[i] for i in indices.reshape(indices.shape[1])]
  anime_list.pop(0)
  return anime_list

def get_animeid(name: str):
    anime_details = animelist[animelist["English name"].str.contains(name.lower(), case=False) | animelist["Name"].str.contains(name.lower(), case=False)]
    if (len(anime_details) == 0):
        return -1
    else:
        id_anime = anime_details.at[anime_details.index[0], 'MAL_ID']
        return id_anime

def start(update, context):
    user = update.effective_user
    update.message.reply_text(
        f"""Hi {user['first_name']} {user['last_name']}!, I am an anime recommendation bot. Please tell me your favourite anime with the command "like" so i can find animes related to that!."""
    )

def animerec(update, context):
    text = ' '.join(context.args)
    anime_id = get_animeid(text)
    if (anime_id == -1):
        s = "There is no anime with such name!"
    else:
        ids = get_knnrecommendation(knnmodel, anime_id, sparse_y)
        anime_list = get_animenames(ids)
        s = f"Animes related to {text} are:"
        for i in range(len(anime_list)):
            s += f"\n{i+1}. {anime_list[i][0]} | {anime_list[i][1]}"

    update.message.reply_text(
        s
    )

def main():
    updater = Updater(BOT_TOKEN)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("like", animerec))

    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=BOT_TOKEN,
                          webhook_url='https://animerecbot.herokuapp.com/' + BOT_TOKEN)

    updater.idle()

if __name__ == '__main__':
    main()