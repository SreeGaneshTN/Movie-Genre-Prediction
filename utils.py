import requests
import os


def poster_download(data,file_destination):
    for index,row in data.iterrows():
        img_path=str(file_destination)+'/'+str(row.imdbId)+'.jpg'
        link=str(row.Poster)
        r=requests.get(link)
        with open(img_path ,'wb') as f:
            f.write(r.content)
        

