import sys
import numpy as np


def main(argv):
    print "\n"
    input_file = argv[1]
    # print 'name input file:', input_file

    training_examples = np.loadtxt(input_file)
    list_of_train = training_examples.tolist()
    xy_num_of_col = len(list_of_train)
    x_num_of_row = len(list_of_train[0])-1

    y_vector = []
    index_ofy = len(list_of_train[0])-1
    for i in range(xy_num_of_col):
        y_vector.append(list_of_train[i][index_ofy])

    temp_x_matrix = []
    temp_list = []
    for col in range(xy_num_of_col):
        for row in range(x_num_of_row):
            temp_list.append(list_of_train[col][row])
        temp_x_matrix.append((list(temp_list)))
        temp_list = []

    x_matrix = []
    temp_x = []
    for row in range(x_num_of_row):
        for col in range(xy_num_of_col):
            temp_x.append(temp_x_matrix[col][row])
        x_matrix.append((list(temp_x)))
        temp_x = []

    # consistnty algorithm

    # init hypothesis
    init_hypothesis = []
    temp_var_name = []
    for i in range(x_num_of_row):
        temp_var_name = "x" + str(i+1)
        init_hypothesis.append(temp_var_name)
        temp_var_name = "not(x" + str(i+1) + ")"
        init_hypothesis.append(temp_var_name)
        temp_var_name = []

    num_of_instance = xy_num_of_col

    last_hypothesis = init_hypothesis
    for myval in range(num_of_instance):
        i = myval + 1
        current_hypothesis = last_hypothesis
        index_for_y = i-1
        y_iteration_t = y_vector[index_for_y]

        list_our_predict = []
        for k in range(len(current_hypothesis)):
            current_word = current_hypothesis[k]
            number_of_x= []
            if "not" in current_word:
                number_of_x = current_word[5]
                temp = x_matrix[int(number_of_x)-1]
                number_of_x_matrix = 0
                value_in_matrix = temp[number_of_x_matrix]
                if value_in_matrix == 0.0:
                    list_our_predict.append(1.0)
                else:
                    list_our_predict.append(0.0)
            else:
                number_of_x = current_word[1]
                temp = x_matrix[int(number_of_x)-1]
                number_of_x_matrix = 0
                value_in_matrix = temp[number_of_x_matrix]
                if value_in_matrix == 0.0:
                    list_our_predict.append(0.0)
                else:
                    list_our_predict.append(1.0)
        is_zero = 0  # if 0 we didnt find zero, else 1 is_--->zero=y gag
        for j in range(len(list_our_predict)):
            if is_zero == 0:
                    if list_our_predict[j] == 0:
                        is_zero = 1



        # if y t = 1 and y gag t = 0 (our hypothesis is no good any more) then
        if (y_iteration_t == 1) & (is_zero == 1):
            current_instance = []
            for f in range(int(x_num_of_row)):
                temp = x_matrix[f]
                number_of_x_matrix = myval
                value_in_matrix = temp[number_of_x_matrix]
                current_instance.append(value_in_matrix)

            #  for index i in instance t do
            for m in range(len(current_instance)):
                current_check = current_instance[m]

                if current_check == 1.0:
                    number_to_remove = "not(x" + str(m+1) + ")"
                    if number_to_remove in current_hypothesis:
                        current_hypothesis.remove(number_to_remove)
                else:
                    if current_check == 0.0:
                        number_to_remove = "x" + str(m+1)
                        if number_to_remove in current_hypothesis:
                            current_hypothesis.remove(number_to_remove)

        if y_iteration_t == 0:
            last_hypothesis = current_hypothesis

    string_answer =""
    for r in range(len(last_hypothesis)):
        string_answer += last_hypothesis[r]
        if r != len(last_hypothesis) - 1:
            string_answer += ","

    file = open('output.txt', 'w')
    file.write(string_answer)
    file.close()
    pass

if __name__ == "__main__":
    main(sys.argv)