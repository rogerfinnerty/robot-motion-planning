"""
A pedagogical implementation of a priority queue
"""

from numbers import Number


class PriorityQueue:
    """ Implements a priority queue """
    def __init__(self):
        """
        Initializes the internal attribute  queue to be an empty list.
        """
        self.queue_list = []

    def check(self):
        """
        Check that the internal representation is a list of (key,value) pairs,
        where value is numerical
        """
        is_valid = True
        for pair in self.queue_list:
            if len(pair) != 2:
                is_valid = False
                break
            if not isinstance(pair[1], Number):
                is_valid = False
                break
        return is_valid

    def insert(self, key, cost):
        """
        Add an element to the queue.
        """
        self.queue_list.append( (key, cost) )

    def min_extract(self):
        """
        Extract the element with minimum cost from the queue.
        """
        if len(self.queue_list) == 0:
            key = None
            cost = None
        else:
            min_idx = 0
            for i in range(1, len(self.queue_list) ):
                if self.queue_list[i][1] < self.queue_list[min_idx][1]:
                    min_idx = i
            key, cost = self.queue_list[min_idx]

        return key, cost

    def is_member(self, key):
        """
        Check whether an element with a given key is in the queue or not.
        """
        flag = False
        for _, item in enumerate(self.queue_list):
            if item[0] == key:
                flag = True
        return flag

    def print_queue(self):
        """
        Print contents of the queue 
        """
        if len(self.queue_list) == 0:
            print("Queue is empty.")
        else:
            print("Current queue: ")
            for item in self.queue_list:
                print(f"Key: {item[0]}, Cost: {item[1]}.")

    def display_descending(self):
        """
        Displays the elements of the queue in descending order by cost. 
        """
        # First create a duplicate priority queue so that we can delete elements as they are printed
        queue_copy = PriorityQueue()
        for _, item in enumerate(self.queue_list):
            key = item[0]
            cost = item[1]
            queue_copy.insert(key, cost)

        # Modified version of extract_min function to find the index
        # of the maximum cost item, print that item,
        # then remove it from the list
        for _ in range(len(self.queue_list)):
            # Determine index of maximum value
            max_idx = 0
            for j in range(1, len(queue_copy.queue_list) ):
                if queue_copy.queue_list[j][1] > queue_copy.queue_list[max_idx][1]:
                    max_idx = j
            # Display the maximum key, value pair
            max_key = queue_copy.queue_list[max_idx][0]
            max_val = queue_copy.queue_list[max_idx][1]
            print(f"Key: {max_key},Cost: {max_val}")
            # Remove the maximum pair from the list
            queue_copy.queue_list.pop(max_idx)
