from models import SAO, matching, sao_distance
from udpipe import UDPipeParser
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import row_number, lit
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from lxml import etree
import re
import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict, Counter
from nltk import sent_tokenize
MODEL = "/home/vagrant/english-ewt-ud-2.5-191206.udpipe"

#Формирует уникальный идентификатор патента
#patent_data - текст с xml-разметкой патента
def get_uniq_id(patent_data):
    root = etree.fromstring(patent_data.encode('utf-8'))
    doc_numers = root.findall(".//doc-number")
    countries = root.findall(".//country")
    kind = root.findall(".//kind")
    uniq = countries[0].text + doc_numers[0].text + kind[0].text
    return uniq

#Получает название патента
#patent_data - текст с xml-разметкой патента
def get_patent_name(patent_data):
    root = etree.fromstring(patent_data.encode('utf-8'))
    patent_name = root.find(".//invention-title").text
    return patent_name

#Получает страну, выдавшую патент
#patent_data - текст с xml-разметкой патента
def get_country(patent_data):
    root = etree.fromstring(patent_data.encode('utf-8'))
    countries = root.findall(".//country")
    country_name = countries[0].text
    return country_name

#Получает номер патента
#patent_data - текст с xml-разметкой патента
def get_number(patent_data):
    root = etree.fromstring(patent_data.encode('utf-8'))
    doc_numers = root.findall(".//doc-number")
    number = doc_numers[0].text
    return number

#Получает дату патента
#patent_data - текст с xml-разметкой патента
def get_date(patent_data):
    root = etree.fromstring(patent_data.encode('utf-8'))
    date_patent = root.findall(".//date")
    date = date_patent[0].text
    return date

#Получает текст патента
#patent_data - текст с xml-разметкой патента
def get_description(patent_data):
    root = etree.fromstring(patent_data.encode('utf-8'))
    descriptions = root.findall(".//p")
    if (descriptions):
        description_without_tags = [''.join(description.itertext()) for description in descriptions]
        return ' '.join(description_without_tags)
    return ''

#Предобработка текста патента
#text - текст патента
def preparing_text(text):
    text = re.sub(r'(FIG.|FIGS.|No. \d+-\d+|\d+/\d+|\d+)(,)?\s*', r'', text)
    text = re.sub(r'U\.S\.', r'US', text)
    text = re.sub(
        r'(\s*)(, wherein|, said|, and|; and|, thereby|if|else|thereby|such that|so that|wherein|whereby|where|when|while|but|:|;)',
        r'.', text)
    text = re.sub(
        r'(\.\s+|^)(\d{1,4}|[a-zA-Z]{1,2})(\.|\))',
        r'. ', text)
    text = re.sub(
        r'\.\s.+(of|in|to) claim \d+(, )?',
        r'. ', text)
    text = re.sub(
        r'(\s)?\[.*\](\s)?', r'', text)
    text = text.replace(r'“', '')
    text = text.replace(r'”', '')
    paragraphs = text.split('\n')
    paragraphs = map(lambda x: x.strip(), paragraphs)
    paragraphs = sent_tokenize('\n'.join(paragraphs))
    paragraphs = filter(lambda x: len(x) > 30, paragraphs)
    text = ' '.join(paragraphs)
    text = text.replace("\n", " ")
    return text

#Формирует список SAO из текста патента
#text - предоброботанный текст патента
def get_sao(text):
    parser = UDPipeParser(MODEL)
    trees = parser.parse(text)
    sao_list = chain(*map(SAO.extract, trees))
    sao_spisok = []
    for sao in sao_list:
        sao_spisok.append(sao)
    return np.array(sao_spisok)

def sao(sao_list):
    sao_spisok = []
    for sao in sao_list:
        sao_spisok.append(str(sao))
    return sao_spisok

#Группирует данные для расчета tf-idf
#matches - сгруппированные SAO по степени схожести (словарь: key - номера групп, value - списки SAO)
#pat_sao - словарь: key - патенты, value - списки SAO
def to_group(matches, pat_sao):
    sao_group = {sao: group
                 for group, sao_list in matches.items()
                 for sao in sao_list}
    pat_gsao = defaultdict(list)
    for pat, sao_list in pat_sao.items():
        groups = [sao_group[sao] for sao in sao_list]
        pat_gsao[pat] = groups
    groups_count = len(matches.keys())
    pat_group_list = defaultdict(list)
    for pat, group_list in pat_gsao.items():
        counter = Counter(group_list)
        pat_group_list[pat] = [counter[index] for index in range(groups_count)]
    return pat_group_list

#Вывод данных по кластерам и сохранение их в БД
#center - центры кластеров
#vocab_frame - dataframe, содержащий все извлеченные SAO
#column - массив уникальных SAO
#title_frame - dataframe, содержащий названия патентов
def topic(center, vocab_frame, column, title_frame):
    client = MongoClient('localhost', 27017)
    db = client['PatentDB']
    sao_collection = db['sao']
    topic_sao = []
    print("Top SAO per cluster:")
    print()
    order_centroids = center.argsort()[:, ::-1]
    for i in range(3):
        topic_sao.clear()
        print("%d сluster SAO:" % (i + 1), end='')
        print()
        for ind in order_centroids[i, :20]:
            print(vocab_frame.loc[column[ind]].values.tolist(), end=' ')
            topic_sao.append(str(vocab_frame.loc[column[ind]].values.tolist()))
        print()
        print("%d сluster titles:" % (i + 1), end='')
        print()
        for title in title_frame.loc[i]['title'].values.tolist():
            print(' %s,' % title, end='')
            print()
        print()
        sao_collection.save(
            {"cluster": i, "sao": topic_sao, "patent": title_frame.loc[i]['title'].values.tolist()})
    print("_" * 10)
    client.close()

#Сохраняет данные парсинга патентов в БД
#prepared_df - dataframe, содержащий ключевые поля патента
def save_patent_to_DB(prepared_df):
    client = MongoClient('localhost', 27017)
    db = client['PatentDB']
    patents_collection = db['patents']
    for row in prepared_df.collect():
        flag = 0
        for pat in patents_collection.find():
            if pat["_id"] == row.uniq:
                flag = 1
                break
        if flag == 0:
            patents_collection.save(
                {"_id": row.uniq, "name": row.patent_name, "country": row.country, "number": row.number,
                 "date": row.date, "text": row.patent_description})
    client.close()

spark = SparkSession \
    .builder \
    .appName("Application") \
    .getOrCreate()

input_data = spark.sparkContext.wholeTextFiles('/home/vagrant/spark_app/mon/*.xml')

#Парсинг патентов
month = "february"
prepared_data = input_data.map(lambda x: (get_uniq_id(x[1]), get_patent_name(x[1]), get_country(x[1]), get_number(x[1]), get_date(x[1]), get_description(x[1]))) \
    .map(lambda x: (x[0], x[1], x[2], x[3], x[4], preparing_text(x[5])))
prepared_df = prepared_data.toDF().selectExpr('_1 as uniq', '_2 as patent_name', '_3 as country', '_4 as number', '_5 as date', '_6 as patent_description')
prepared_df.show()
if (month == "march"):
    prepared_df = prepared_df.filter(prepared_df["date"].rlike(r'202003\d+'))
if (month == "february"):
    prepared_df = prepared_df.where(prepared_df["date"].rlike(r'202002\d+'))
w = Window.orderBy(lit('A'))
prepared_df = prepared_df.withColumn('id', (row_number().over(w) - 1))
prepared_df.show()
save_patent_to_DB(prepared_df)

#Извлечение SAO-структур
sao_data = prepared_df.rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], get_sao(x[5])))
sao_prepared = sao_data.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], sao(x[7])))
sao_df = sao_prepared.toDF().selectExpr('_1 as uniq', '_2 as patent_name', '_3 as country', '_4 as number', '_5 as date', '_6 as patent_descrition', '_7 as id', '_8 as sao')
sao_df.show()
patents = sao_data.map(lambda x: (x[0], x[7])).collect()
total_vocab_sao = []
pat_sao = defaultdict(list)
sao_pat = defaultdict(list)
for pat, sao_list in patents:
    pat_sao[pat].extend(sao_list)
    total_vocab_sao.extend(sao_list)
    for sao in sao_list:
        sao_pat[sao].append(pat)
        print(sao._tree)
        print("_" * 10)
vocab_frame = pd.DataFrame({'sao': total_vocab_sao}, index=total_vocab_sao)
total_sao_list = list(chain(sao_pat.keys()))
column = np.array(total_sao_list)

#Расчет частотных характеристик
matches = matching(sao_pat.keys(), sao_distance, 0.8)
pat_group_list = to_group(matches, pat_sao)
tf_mtx = [groups for pat, groups in pat_group_list.items()]
tf = np.array(tf_mtx)
idf = np.sum(np.where(tf > 0, 1, 0), axis=0) / len(patents)
tf_idf = tf * idf

#Сохранение tf-idf в dataframe
featurized_data = sao_df.rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], Vectors.dense(list(tf_idf[x[6]]))))
featurized_df = featurized_data.toDF().selectExpr('_1 as uniq', '_2 as patent_name', '_3 as country', '_4 as number', '_5 as date', '_6 as patent_descrition', '_7 as id','_8 as sao', '_9 as features')
featurized_df.show()
featurized_df.select('features').show(truncate=False, vertical=True)

#Кластеризация патентов
kmeans = KMeans(k=3, seed=1, distanceMeasure="cosine")
model = kmeans.fit(featurized_df.select('features'))
clusters = model.transform(featurized_df)
clusters.show()
centers = model.clusterCenters()
center = np.array(centers)

#Расчет косинусного расстояния
mat = IndexedRowMatrix(
    featurized_df.select('id', 'features')\
        .rdd.map(lambda row: IndexedRow(row.id, row.features.toArray()))).toBlockMatrix()
dot = mat.multiply(mat.transpose())
dist = dot.toLocalMatrix().toArray()
cluster_array = [int(row.prediction) for row in clusters.select('prediction').collect()]
pat_array = [row.patent_name for row in featurized_df.select('patent_name').collect()]
uniq_array = [row.uniq for row in featurized_df.select('uniq').collect()]
title_frame = pd.DataFrame({'title': pat_array}, index=[cluster_array])

#Вывод списка SAO и названий патентов по кластерам
topic(center, vocab_frame, column, title_frame)

#Формирование графика
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]
cluster_colors = {0: '#3333CC', 1: '#FFFF00', 2: '#00FF00'}
df = pd.DataFrame(dict(x=xs, y=ys, prediction=cluster_array, title=uniq_array))
# Группируем кластеры
groups = df.groupby('label')
# Настраиваем вывод
fig, ax = plt.subplots(figsize=(15, 7))  # Устанавливаем размер - длину и ширину окна
ax.margins(0.05)  # добавляем 5% заполнения к автомасштабированию (не обязательно)
# Используем имена и цвета кластеров с поиском 'name', чтобы вернуть соответствующий цвет/метку
indx = 0
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # изменения применяются к оси х
        which='both',  # затронуты как основные, так и второстепенные галки
        bottom='off',  # галочки по нижнему краю сняты
        top='off',  # галочки по верхнему краю сняты
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # изменения применяются к оси y
        which='both',  # затронуты как основные, так и второстепенные галки
        left='off',  # галочки по нижнему краю сняты
        top='off',  # галочки по верхнему краю сняты
        labelleft='off')
    indx += 1
ax.legend(numpoints=1)

# Добавление точки в позиции x и y с пометкой названия документа
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)
plt.show()  # Вывод ландшафта

spark.stop()

"""
from pymongo import MongoClient

# Create the client
client = MongoClient('localhost', 27017)

# Connect to our database
db = client['Patent']

# Fetch our series collection
series_collection = db['patents']

def insert_document(collection, data):
    #Function to insert a document into a collection and return the document's id.
    return collection.insert_one(data).inserted_id


def find_document(collection, elements, multiple=False):
    """ #Function to retrieve single or multiple documents from a provided
#Collection using a dictionary containing a document's elements.
"""
    if multiple:
        results = collection.find(elements)
        return [r for r in results]
    else:
        return collection.find_one(elements)

def update_document(collection, query_elements, new_values):
    #Function to update a single document in a collection.
    collection.update_one(query_elements, {'$set': new_values})

def delete_document(collection, query):
    #Function to delete a single document from a collection.
    collection.delete_one(query)

# new_show = {
#     "name": "FRIENDS",
#     "year": 1994
# }

#print(insert_document(series_collection, new_show))
# result = find_document(series_collection, {'name': 'FRIENDS'})
# print(result)
new_show = {
    "name": "FRIENDS",
    "year": 1995
}
id_ = insert_document(series_collection, new_show)
# update_document(series_collection, {'_id': id_}, {'name': 'F.R.I.E.N.D.S'})
# result = find_document(series_collection, {'_id': id_})
# print(result)

delete_document(series_collection, {'_id': id_})
result = find_document(series_collection, {'_id': id_})
print(result)
"""


# from models import SAO
# from udpipe import UDPipeParser
# from itertools import chain, islice
# import re
# MODEL = "/home/vagrant/english-ewt-ud-2.5-191206.udpipe"
# TEXT = """
#     The super-capacitor electrode further comprises a silane coupling agent. The electrode comprises a silane coupling agent.
# """
# string = """
#     1. i love you (ex nothing) (1056). 4. The device according to claim 3, the super-capacitor electrode further comprises a silane coupling agent. 1) A super-capacitor electrode comprising a metal foil as a current collector, an active material, a conductive agent and an organic adhesive agent, wherein the super-capacitor electrode further comprises a silane coupling agent for binding the organic adhesive agent and the uncorroded smooth metal foil, and The electrode comprises a silane coupling agent.
# """
# #A super-capacitor electrode comprising a metal foil as a current collector, an active material, a conductive agent and an organic adhesive agent, wherein the metal foil is an uncorroded smooth metal foil, and the super-capacitor electrode further comprises a silane coupling agent for binding the organic adhesive agent and the uncorroded smooth metal foil so that the active material is adhered to the uncorroded smooth metal foil.
# if __name__ == "__main__":
#     str = re.sub(
#         r'(\s*)(, wherein|, said|, and|; and|, thereby|if|else|thereby|such that|so that|wherein|whereby|where|when|while|but|:|;)',
#         r'.', string)
#     str = re.sub(
#         r'(\s+|^)(\d{1,4}|[a-zA-Z]{1,2})(\.|\))',
#         r'', str)
#     str = re.sub(
#         r'\.\s.+(of|in|to) claim \d+(, )?',
#         r'. ', str)
#     str = re.sub(
#         r'\s\(.+\)',
#         r'', str)
#     print(str)
#     parser = UDPipeParser(MODEL)
#     trees = parser.parse(str)
#     sao = chain(*map(SAO.extract, trees))
#     first, second, three = islice(sao, 3)
#     #print("K:", first.compare(second))
#     print("First:")
#     print(first._tree)
#     print("Second:")
#     print(second._tree)
#     print("three:")
#     print(three._tree)