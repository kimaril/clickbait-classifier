import os
import pandas as pd
import json

def to_files(dataframe, directory, id_list, content):
    for i in id_list:
        filename = "{0}/{1}.json".format(directory, i)
        with open(filename, mode='w') as f:
            dataframe.loc[i][content].to_json(f, force_ascii=False,  orient='records', lines=True)

def to_dirs(dataframe, rootdir, col="label", content="text"):  
    """dataframe: откуда раскидываем по директориям
    directory: название корневой папки
    col: по какой колонке формируем директории; datafarme[col].unique()
    content: что пишем в файлы"""
    
    if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        
    categories = dataframe[col].unique()
    
    for cat in categories:
        directory = "{0}/{1}".format(rootdir, cat)
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        id_list = dataframe[dataframe[col]==cat].index.tolist()
        print("Writing to {0} directory...".format(cat))
        to_files(dataframe=dataframe, directory=directory, id_list=id_list, content=content)