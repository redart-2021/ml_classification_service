# ml_classification_service
## В основе работы сервиса лежит модель машинного обучения на основе KNeighborsClassifier
Запуск
-
~~~
pip3 install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 wsgi:app
~~~
## Request
## GET /
## Response
~~~
{"Service status":"OK!","used model":"core/models/knn_model.model"}
~~~
### Request
#### POST /predict
~~~
{
   "right_fields": 900,
   "wrong_fields": 4,
   "user_stats": 27.51388888888889
}
~~~
### Response
~~~
{
    "user_type": "3"
}
~~~
