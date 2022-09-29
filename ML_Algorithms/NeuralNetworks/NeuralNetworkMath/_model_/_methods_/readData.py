def readData(path: str) -> list:
    data_file = open(path)
    data_list = data_file.readlines()
    data_file.close()
    return data_list
