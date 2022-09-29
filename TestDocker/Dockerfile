# имя базового образа
FROM python

MAINTAINER Kropotov Nikita
# Путь до рабочей директории внутри образа
WORKDIR /image_scripts
# Копирование локальныйх файлов из текущей папки (.) в папку image_scripts (.) в образе
COPY . .
EXPOSE 8002
#RUN pip install pystrich
# Запуск команды, которая выполнится внутри контейнера, который будет создан на основе образа
CMD [ "python", "Main.py"]

