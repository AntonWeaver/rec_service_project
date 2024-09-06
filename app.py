from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from loguru import logger
from pydantic import BaseModel
import os
import hashlib

#from schema import PostGet

SQLALCHEMY_DATABASE_URL = "XXXX"
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=10, max_overflow=-1)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_model_path(path: str, model_ver: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = f'/workdir/user_input/model_{model_ver}'
    else:
        MODEL_PATH = f'{path}_{model_ver}'
    return MODEL_PATH

def load_models(model_ver: str):
    #model_path = get_model_path("/my/super/path") #для проверки, локально код ниже
    #model_path = get_model_path("E:/Документы/ML_course/Project_22_actual/model")

    model_path = get_model_path('E:/Документы/ML_course/Project_22_stats/model', model_ver)
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://XXX"
        "postgresXXX"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features(sql_query: str) -> pd.DataFrame:
    result = batch_load_sql(sql_query)
    return result

#В зависимости от пользователя подгружаем таблицы (разные под разные модели)

user_prep = load_features("""SELECT * FROM tkachenko_a_user_features__lesson_22 """)
post_prep_control = load_features("""SELECT * FROM tkachenko_a_post_features__lesson_22 """)
post_prep_test = load_features("""SELECT * FROM tkachenko_a_post_mod_features__lesson_22 """)

###ENDPOINT etc.###

app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db

#Для локальной работы
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True
#Для локальной работы
class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

#Для локальной работы
class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

#Загружаем две модели (контрольную и тестовую
model_control = load_models('control')
model_test = load_models('test')

#Сплит пользователей

salt = "my_salt"

def get_exp_group(user_id: int) -> str:
    value_str = str(user_id) + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num %100
    if percent < 50:
        return 'control'
    elif percent < 100:
        return 'test'
    return 'unknown'

def get_rec_for_exp(id: int, limit: int, user_group: str, user_prep: pd.DataFrame, post_prep: pd.DataFrame, model: None, db = None) -> List[Response]:

    user_qr = user_prep[user_prep['user_id'] == id].drop('user_id', axis =1)
    features_prep = pd.concat((user_qr, post_prep.drop(['text'], axis=1)), axis=1).fillna(method='ffill')

    features_prep['prediction'] = model.predict_proba(features_prep.drop('post_id', axis=1))[:, 1]

    post_rec = tuple(features_prep.sort_values('prediction', ascending=False)['post_id'].head(limit))

    result = (db.query(Post.id, Post.text, Post.topic)
              .filter(Post.id.in_(post_rec)).all()
              )
    full_result = {'exp_group': user_group, 'recommendations': result}

    if result is None:
        raise HTTPException(200, [])
    else:
        return full_result


@app.get("/post/recommendations/", response_model=Response)
def get_rec(id: int = None, limit: int = 5, db: Session = Depends(get_db)) -> List[Response]:

    user_group = get_exp_group(user_id = id)
    logger.info(f'user group - {user_group}')

    if user_group == 'control':
        model = model_control
        post_prep = post_prep_control

    elif user_group == 'test':
        model = model_test
        post_prep = post_prep_test
    else:
        raise ValueError('group is unknown')

    return get_rec_for_exp(id, limit, user_group, user_prep, post_prep, model, db)
