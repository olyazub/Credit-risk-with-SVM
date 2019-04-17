import random

import matplotlib




def main_K_means(data_frame):
    centr = get_random_centroid(data_frame)
    k_means_one_it(data_frame, centr)


def get_random_centroid(data, n=1):
    if n > len(data):
        return False
    centr1 = data[0]
    centr2 = data[n]
    if centr1 == centr2:
        get_random_centroid(data, n + 1)
    else:
        return [centr1, centr2]


def k_means_one_it(data, centr):
    getLabels(data, centr)
    new_centroids = cal_new_centr(data)
    print(' ')
    print(centr)
    print(new_centroids)

    if centr[0] in new_centroids and centr[1] in new_centroids:
        print('Data is classified')
        print(data)
        return True
    else:
        print('New iteration')
        return k_means_one_it(data, new_centroids)


def length(point, centroid):
    length = 0;
    for i in range(len(point) - 1):
        length += (centroid[i] - point[i]) ** 2;
    return length ** 0.5


def getLabels(dataSet, centroids):
    for point in dataSet:
        if length(point, centroids[0]) > length(point, centroids[1]):
            point[-1] = 0
        else:
            point[-1] = 1


def cal_new_centr(data):
    '''Here I calculate new centroids as average '''
    sum_1 = []
    sum_2 = []
    n = len(data[0]) - 1
    for i in range(n):
        sum_1.append(0)
        sum_2.append(0)

    n_centr1 = 0
    for i in data:
        if i[-1] == 0:
            n_centr1 += 1
            for j in range(len(i) - 1):
                sum_1[j] = sum_1[j] + i[j]
        else:
            for j in range(len(i) - 1):
                sum_2[j] = sum_2[j] + i[j]

    n_centr2 = len(data) - n_centr1

    for l in range(len(sum_1)):
        sum_1[l] = sum_1[l] / n_centr1
        sum_2[l] = sum_2[l] / n_centr2

    return [sum_1, sum_2]


if __name__ == "__main__":
    points = []
    for i in range(30):
        points.append([random.randint(0, 101), random.randint(0, 101), None])
    print(points)
    main_K_means(points)
