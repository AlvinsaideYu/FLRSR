def write_to_log(data_list, file_name):
    try:
        with open(file_name, 'a') as file:
            line = ' '.join(str(data) for data in data_list)  # 使用空格将数据连接成一行
            file.write(line + '\n')  # 写入一行数据并换行
        print("数据已记录到文件:", file_name)
    except Exception as e:
        print("写入数据时出错:", str(e))

# 示例数据
data_list = [1, 2, 3, 4, 5]  # 这里假设每组数据有五个整数
log_file_name = "data_log.txt"
write_to_log(data_list, log_file_name)
