import sqlite3
import datetime
from datetime import datetime


def get_timestamp(date):
    dt_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    millisec = dt_obj.timestamp() * 1000
    return millisec


insert_data = [
    (1, "2016-01-01 09:08:53", "Альфабанк", 111),
    (2, "2016-01-02 15:09:34", "Альфабанк", 112),
    (3, "2016-01-03 12:45:23", "ХоумКредит", 113),
    (4, "2016-07-23 17:08:22", "Sber", 111),
    (5, "2016-12-06 14:08:53", "ХоумКредит", 112),
    (6, "2016-08-10 12:11:13", "Альфабанк", 113),
    (7, "2016-11-31 15:01:23", "Тинькоф", 114),
    (8, "2017-08-10 12:11:13", "Альфабанк", 113),
]

# with sqlite3.connect('TestDataBase/database.db') as db:
#     cursor = db.cursor()
#     query_create = """ CREATE TABLE IF NOT EXISTS users (Applicat_ID INTEGER, Applicat_Date TEXT, Bank_Name TEXT, Passport_Number INTEGER) """
#     query_add = """ INSERT INTO users(Applicat_ID, Applicat_Date, Bank_Name, Passport_Number) VALUES(?,?,?,?) """
#     cursor.execute(query_create)
#     cursor.executemany(query_add, insert_data)
#     db.commit()

with sqlite3.connect('TestDataBase/database.db') as db:
    cursor = db.cursor()
    query_get = """
   select distinct Passport_Number
   from(
   select Passport_Number, 
          lag(Bank) over(partition by Passport_Number) - Bank as diff
   from(
   select Passport_Number, 
          Applicat_Date, 
          case when Bank_Name = 'Альфабанк' then 5
               when Bank_Name = 'ХоумКредит' then 2
               else 0
          end as Bank 
   from users 
   order by Passport_Number, Applicat_Date
   )
   )
   where diff = 3
    """
    cursor.execute(query_get)
    for res in cursor:
        print(res)
    db.commit()
