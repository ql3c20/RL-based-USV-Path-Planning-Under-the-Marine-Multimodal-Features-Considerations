def read_binary_txt(file_path):
    matrix_list = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(value) for value in line.strip().split()]
            matrix_list.append(row)
    return matrix_list

