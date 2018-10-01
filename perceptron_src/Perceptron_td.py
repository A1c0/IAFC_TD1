from .Neuron import Neuron
import numpy as np
import random
import pickle

class Perceptron(object):

    def __init__(self, numberOfNeurons, imageWidth, imageHeight, weights=None):
        """Loads an existing neural network if weights points to a valid 
           weights file, otherwise, initialises a new neural network.
        
        Arguments:
            numberOfNeurons {int} -- The number of neurons in the network
            imageWidth {int} -- The width  of the input images in pixels
            imageHeight {int} -- The height of the input images in pixels
        
        Keyword Arguments:
            weights {string} -- The path to the weights file, or None to 
                                initialize a new network(default: {None})
        """

        if weights is None:
             # TODO Exercise 1 Initalize a new network containing
             # numberOfNeurons neurons each with a different position chosen
             # at random
            self.network = []
            list = self.generatePositionList(imageWidth, imageHeight)
            for i in range(numberOfNeurons):
                while True:
                    neutron = Neuron(random.choice(list))
                    if neutron not in self.network:
                        self.network.append(neutron)
                        break
        else:
            # TODO Exercise ? Load an existing network
            self.load(weights)


    def __eq__(self, other):
        return (self.network == other.network)
    
    
    def generatePositionList(self, imageWidth, imageHeight):
        """Generate a list of all possible positions in an image of dimensions
            imageWidth x imageHeight
        
        Arguments:
            imageWidth {int} -- The width of the image
            imageHeight {int} -- The height of the image
        
        Returns:
            list((tuple(int,int))) -- The list of positions presented as (x, y)
        """

        positionList = []
        for x in range(imageWidth):
            for y in range(imageHeight):
                positionList.append((x,y))
        
        return positionList

    # TODO Exercise 2: Implement the forward pass function
    def forwardPass(self, image):
        """Takes a binary image as input, computes the weighted sum of the
            of the products each neuron weight with the associated pixel value 
            and activates the neuron if the pixel value is 1.
            It then returns -1 if the weighted sum is negative, 0 if the 
            weighted sum is 0 and +1 if the weighted sum is positive.
        
        Arguments:
            image {numpy.array} -- The binary image input
        """

        result = 0
        for neutron in self.network:
            neutron.active = True if image[neutron.yPos, neutron.xPos] == 1 else False
            result += neutron.value * image[neutron.yPos, neutron.xPos]
        if result > 0:
            return 1
        elif result < 0:
            return -1
        else:
            return 0

    # TODO Exercise 3: Implement the backpropagation function
    def backProp(self, expectedResult, result):
        """If the expected result does not match the actual result, add the 
           expected result to the value of all active neurons, then deactivate
           them.
        
        Arguments:
            expectedResult {int} -- The expected result for the forward pass
            result {int} -- The actual result for the forward pass
        """
        
        pass

    # TODO Exercise 4: Implement the error calculation function
    def calcError(self, labels, results):
        """Compute the number of label/result pairs that are not equal and 
           return the result.
        
        Arguments:
            labels {list(int)} -- The list of labels
            results {list(int)} -- The list of results
        """

        return 0


    # TODO Exercise 5: Implement the training function
    def train(self, images, labels, maxIterations):
        """Train the perceptron by computing the result of a forward pass then 
            by backpropagating that result through the neural network until the
            error reaches 0 or the max iteration number has been reached.
        
        Arguments:
            images {[type]} -- [description]
            labels {[type]} -- [description]
            numIterations {[type]} -- [description]
        """
        
        pass

    # TODO Exercise 6: Implement the testing function
    def test(self, images):
        """Compute the forward pass results for a list of images
        
        Arguments:
            images {numpy.array} -- A list of binary images
        
        Returns:
            list(int) -- The forward pass results for each input image
        """

        return []

    # TODO Bonus: write the save and load functions that will allow you to save
    # the result of a training and reuse those weights later
    def save(self, file):
        """Save the current neural network to a file using 
            pickle
        
        Arguments:
            file {str} -- The path to the file the weights will be saved to
        """

        pass


    def load(self, file):
        """Load the neural network from a pickle file 
        
        Arguments:
            file {str} -- The path to the file containing the weights
        """

        self.network = []
