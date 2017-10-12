# Name: Hieu Le - htl5683@truman.edu

# Implements the K-means clustering algorithm.

import matplotlib.pyplot as pyplot
import numpy.random as random


class Point:
    """
    A point on the Cartesian plane is represented by its x and y coordinates
    """
    def __init__(self, x, y):
        """
        Initializes a point on the Cartesian plane from given horizontal and
        vertical coordinates
        """
        self.x = x
        self.y = y

    def get_squared_distance(self, other):
        """
        Returns the square of the Euclidean distance between this other and
        another other
        """
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def __str__(self):
        return '(%f, %f)' % (self.x, self.y)


class KMeansCluster:
    def __init__(self, k, centroids, observations):
        """
        Initialize the K-means clustering algorithm
        :param k: number of partitions
        :param centroids: pre-defined centroids before the first iteration
        :param observations: observations to be partitioned into k clusters
        """
        self.k = k
        self.centroids = centroids
        self.observations = observations
        self.clusters = []

    def assign(self):
        """
        Assigns each object to the cluster with the closest centroid
        """
        self.clusters = [[] for x in range(self.k)]
        for observation in self.observations:
            # Determine index of the closest centroid from observation
            closest_index = 0
            closest_distance = observation.get_squared_distance(
                self.centroids[closest_index])

            for index in range(1, self.k):
                distance = observation.get_squared_distance(
                    self.centroids[index])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_index = index

            print('Assigning point %s to cluster %d'
                  % (observation, closest_index))
            self.clusters[closest_index].append(observation)

    def update(self):
        """
        Updates the centroid of each cluster
        :return: True if there are still some movements to the centroids
        """
        has_movement = False
        for index in range(self.k):
            sum_x, sum_y = 0.0, 0.0
            for observation in self.clusters[index]:
                sum_x += observation.x
                sum_y += observation.y
            centroid = Point(sum_x / len(self.clusters[index]),
                             sum_y / len(self.clusters[index]))
            if self.centroids[index].get_squared_distance(centroid) > 1e-3:
                has_movement = True
            print('Updating cluster %d centroid from %s to %s'
                  % (index, self.centroids[index], centroid))
            self.centroids[index] = centroid
        return has_movement


if __name__ == '__main__':
    num_partitions = 4
    # num_partitions = 6
    num_observations = 1000
    lowerBound = -10.0
    upperBound = 10.0

    initial_centroids = [Point(5, 5), Point(5, -5), Point(-5, 5), Point(-5, -5)]
    initial_centroids.extend([Point(0, 5), Point(0, -5)])
    observations = initial_centroids[:]

    random.seed(22061994)
    x_coordinates = random.uniform(lowerBound, upperBound, num_observations)
    y_coordinates = random.uniform(lowerBound, upperBound, num_observations)
    for i in range(num_observations):
        observations.append(Point(x_coordinates[i], y_coordinates[i]))

    algorithm = KMeansCluster(num_partitions, initial_centroids, observations)
    algorithm.assign()
    while algorithm.update():
        algorithm.assign()

    color_map = ['r', 'g', 'b', 'c', 'm', 'k']
    for i in range(len(algorithm.clusters)):
        for point in algorithm.clusters[i]:
            pyplot.scatter(point.x, point.y, color=color_map[i])
        print(len(algorithm.clusters[i]))

    pyplot.xlim(lowerBound, upperBound)
    pyplot.ylim(lowerBound, upperBound)
    pyplot.show()
