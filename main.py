from perceptron_src.Perceptron_td import Perceptron
from perceptron_src.Neuron import Neuron
import os
import cv2
import colorama
from colorama import Fore
import numpy as np
import random

def testPerceptronInit(numberOfNeurons, imageWidth, imageHeight):
    perceptron = Perceptron(numberOfNeurons, imageWidth, imageHeight)
    outString = ("Testing with numberOfNeurons={}, imageWidth={}, "
                 "imageHeight={}: ".format(numberOfNeurons, imageWidth, imageHeight))
    if len(perceptron.network) != numberOfNeurons:
        outString += ("{}KO| Expected length of network to be {} but "
                      "got {}{}").format(Fore.RED, len(perceptron.network), 
                                         numberOfNeurons, Fore.RESET)
        return outString

    if type(perceptron.network) != list:
        outString += ("{}KO| Expected perceptron.network to be of type {} but "
                      "got {}{}".format(Fore.RED, list, 
                                        type(perceptron.network), Fore.RESET))
        return outString
    
    for netIndex, neuron in enumerate(perceptron.network):
        if type(neuron) != Neuron:
            outString += ("{}KO| Expected all of the objects in "
                          "perceptron.network to be of type {} but the "
                          "object at position {} is of type {}{}").format(Fore.RED,
                                                                        Neuron, 
                                                                        netIndex, 
                                                                        type(neuron),
                                                                        Fore.RESET)
            return outString

    for netIndex, neuron in enumerate(perceptron.network):
        if neuron.xPos not in range(imageWidth) or neuron.yPos not in range(imageHeight):
            outString += ("{}KO| Expected all neuron positions to be within the "
                          "range [(0, 0), ({}, {})[ but the neuron at index {} "
                          "has position ({}, {}){}".format(Fore.RED, imageWidth, 
                                                     imageHeight, netIndex, 
                                                     neuron.xPos, neuron.yPos,
                                                     Fore.RESET))
            return outString

    neuronPositions = []
    for netIndex, neuron in enumerate(perceptron.network):
        if (neuron.xPos, neuron.yPos) in neuronPositions:
            outString += ("{}KO| Expected all neuron positions to be different "
                          "but neurons at index {} and neuron at index {} "
                          "both have position ({}, {}){}".format(Fore.RED, 
                                                                 netIndex,
                                                                 neuronPositions.index((neuron.xPos, neuron.yPos)),
                                                                 neuron.xPos,
                                                                 neuron.yPos,
                                                                 Fore.RESET))
            return outString
        else:
            neuronPositions.append((neuron.xPos, neuron.yPos))

    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString

def testForwardPass(network, image, expected):
    imageHeight, imageWidth = image.shape[:2]    
    perceptron = Perceptron(len(network), imageWidth, imageHeight)
    perceptron.network = network
    result = perceptron.forwardPass(image)
    outString = ("Testing with image:\n{}\nnetwork={}: ".format(image, network))
    if result != expected:
        outString += ("{}KO| Expected {} got {}{}").format(Fore.RED, expected, 
                                                          result, Fore.RESET)
        return outString

    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString

def testBackProp(perceptron, expectedNetwork, expectedResult, result):
    perceptron.backProp(expectedResult, result)
    outString = ("Testing with network={}, expectedResult={}, result={}: "
                    "".format(perceptron.network, expectedResult, result))
    if perceptron.network != expectedNetwork:
        outString += ("{}KO| Expected {} got {}{}").format(Fore.RED, 
                                                           expectedNetwork, 
                                                           perceptron.network, 
                                                           Fore.RESET)
        return outString

    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString

def testCalcError(expectedResult, labels, results):
    perceptron = Perceptron(1,1,1)
    result = perceptron.calcError(labels, results)
    outString = ("Testing with labels={}, results={}: ".format(labels, results))
    if expectedResult != result:
        outString += ("{}KO| Expected {} got {}{}").format(Fore.RED, 
                                                           expectedResult, 
                                                           result, 
                                                           Fore.RESET)
        return outString
    
    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString

def testTraining(perceptron, data, labels):
    perceptron.train(data, labels, 1)
    results=[]
    for image in data:
        results.append(perceptron.forwardPass(image))
    outString = ("Testing with:\nnetwork={},\ndata={},\nlabels={}: "
                    "".format(perceptron.network, data, labels))
    
    if results != labels:
        outString += ("\n{}KO| Tested forward pass after training\nExpected {}"
                      "\nGot      {}{}").format(Fore.RED, 
                                                labels, 
                                                results, 
                                                Fore.RESET)
        return outString

    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString

def testTesting(perceptron, data, expected):
    results = perceptron.test(data)

    outString = ("Testing with:\nnetwork={},\ndata={}: "
                    "".format(perceptron.network, data))

    if results != expected:
        outString += ("\n{}KO| Tested forward pass after training\nExpected {}"
                      "\nGot      {}{}").format(Fore.RED, 
                                                expected, 
                                                results, 
                                                Fore.RESET)
        return outString

    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString

def testSaveAndLoad(perceptron):
    fileName = "saveLoadTest.pkl"
    originalNetwork = list(perceptron.network)
    perceptron.save(fileName)
    outString = ("Testing save and load:")
    if not os.path.isfile(fileName):
        outString += ("\n{}KO| File {} was not created{}").format(Fore.RED, 
                                                                 fileName, 
                                                                 Fore.RESET)
        return outString
    if originalNetwork != perceptron.network:
        outString += ("\n{}KO| Network was modified by save operation:\nWas:{}"
                      "\nIs now:{}{}").format(Fore.RED, originalNetwork, 
                                            perceptron.network, Fore.RESET)
        return outString
    
    perceptron.load(fileName)
    if originalNetwork != perceptron.network:
        outString += ("\n{}KO| Loaded network is different than saved network:"
                      "\nWas:{}\nIs now:{}{}").format(Fore.RED, originalNetwork, 
                                                      perceptron.network, Fore.RESET)
        return outString

    outString += "{}OK{}".format(Fore.GREEN, Fore.RESET)
    return outString
    




def exercise1():
    print("\nExercise 1: Network initialization\n")
    print(testPerceptronInit(1, 1, 1))
    print(testPerceptronInit(2, 2, 2))
    print(testPerceptronInit(3, 8, 8))
    print(testPerceptronInit(64,8,8))



def exercise2():
    print("\nExercise 2: Forward propagation\n")
    network = [Neuron((0,0), 0)]
    image = np.zeros((1,1))
    print(testForwardPass(network, image, 0))
    network = [Neuron((0,0), 1)]
    image = np.zeros((1,1))
    print(testForwardPass(network, image, 0))
    network = [Neuron((0,0), -1)]
    image = np.zeros((1,1))
    print(testForwardPass(network, image, 0))
    network = [Neuron((0,0), 0)]
    image = np.ones((1,1))
    print(testForwardPass(network, image, 0))
    network = [Neuron((0,0), 1)]
    image = np.ones((1,1))
    print(testForwardPass(network, image, 1))
    network = [Neuron((0,0), -1)]
    image = np.ones((1,1))
    print(testForwardPass(network, image, -1))
    network = [Neuron((0,0), -1), Neuron((1,1), 1)]
    image = np.array([[1,0], [0,1]])
    print(testForwardPass(network, image, 0))
    network = [Neuron((0,0), 0), Neuron((1,1), 1)]
    image = np.array([[1,0], [0,1]])
    print(testForwardPass(network, image, 1))
    network = [Neuron((0,0), -1), Neuron((1,1), 0)]
    image = np.array([[1,0], [0,1]])
    print(testForwardPass(network, image, -1))
    network = [Neuron((0,1), -1), Neuron((1,0), 1)]
    image = np.array([[1,0], [0,1]])
    print(testForwardPass(network, image, 0))
    network = [Neuron((0,0), 1), Neuron((1,1), 1)]
    image = np.array([[1,0], [0,1]])
    print(testForwardPass(network, image, 1))
    network = [Neuron((0,0), -1), Neuron((1,1), -1)]
    image = np.array([[1,0], [0,1]])
    print(testForwardPass(network, image, -1))



def exercise3():
    print("\nExercise 3: Back propagation\n")

    perceptron1 = Perceptron(1,1,1)
    perceptron1.network = [Neuron((0,0), 0, False)]
    expectedNetwork = [Neuron((0,0), 0, False)]
    print(testBackProp(perceptron1, expectedNetwork, 1, 0))

    perceptron1.network = [Neuron((0,0), 0, True)]
    expectedNetwork = [Neuron((0,0), 1, False)]
    print(testBackProp(perceptron1, expectedNetwork, 1, 0))

    perceptron1.network = [Neuron((0,0), 0, True)]
    expectedNetwork = [Neuron((0,0), -1, False)]
    print(testBackProp(perceptron1, expectedNetwork, -1, 0))

    perceptron1.network = [Neuron((0,0), 1, True)]
    expectedNetwork = [Neuron((0,0), 0, False)]
    print(testBackProp(perceptron1, expectedNetwork, -1, 1))
    
    perceptron2 = Perceptron(2,2,2)
    perceptron2.network = [Neuron((0,0), -1, False), Neuron((1,1), 1, False)]
    expectedNetwork = [Neuron((0,0), -1, False), Neuron((1,1), 1, False)]
    print(testBackProp(perceptron2, expectedNetwork, -1, 0))

    perceptron2.network = [Neuron((0,0), -1, True), Neuron((1,1), 1, False)]
    expectedNetwork = [Neuron((0,0), -1, False), Neuron((1,1), 1, False)]
    print(testBackProp(perceptron2, expectedNetwork, -1, -1))

    perceptron2.network = [Neuron((0,0), -1, True), Neuron((1,1), 1, True)]
    expectedNetwork = [Neuron((0,0), -2, False), Neuron((1,1), 0, False)]
    print(testBackProp(perceptron2, expectedNetwork, -1, 0))

    perceptron2.network = [Neuron((0,0), -1, True), Neuron((1,1), 1, True)]
    expectedNetwork = [Neuron((0,0), 0, False), Neuron((1,1), 2, False)]
    print(testBackProp(perceptron2, expectedNetwork, 1, 0))

    perceptron2.network = [Neuron((0,0), -1, False), Neuron((1,1), 2, True)]
    expectedNetwork = [Neuron((0,0), -1, False), Neuron((1,1), 2, False)]
    print(testBackProp(perceptron2, expectedNetwork, 1, 1))

    perceptron2.network = [Neuron((0,0), 0, True), Neuron((1,1), 1, True)]
    expectedNetwork = [Neuron((0,0), -1, False), Neuron((1,1), 0, False)]
    print(testBackProp(perceptron2, expectedNetwork, -1, 1))

def exercise4():
    print("\nExercise 4: Error calculation\n")

    print(testCalcError(0, [1], [1]))
    print(testCalcError(1, [1], [0]))
    print(testCalcError(1, [-1], [1]))
    print(testCalcError(0, [1, -1, 1], [1, -1, 1]))
    print(testCalcError(1, [1, -1, 1], [1, -1, 0]))
    print(testCalcError(2, [1, -1, 1], [1, 1, -1]))
    print(testCalcError(3, [1, -1, 1], [0, 0, 0]))


def exercise5():
    print("\nExercise 5: Training\n")

    perceptron = Perceptron(4,2,2)
    images = [np.array([[1, 0],
                        [0, 0]])]
    labels = [1]
    print(testTraining(perceptron, images, labels))

    perceptron = Perceptron(4,2,2)
    images = [np.array([[1, 0],
                        [0, 0]]),
              np.array([[0, 0],
                        [0, 1]])]
    labels = [1, -1]
    print(testTraining(perceptron, images, labels))

    perceptron = Perceptron(4,2,2)
    images = [np.array([[1, 0],
                        [0, 0]]),
              np.array([[0, 0],
                        [0, 1]]),
              np.array([[1, 1],
                        [0, 0]])]
    labels = [1, -1, 1]
    print(testTraining(perceptron, images, labels))

    perceptron = Perceptron(4,2,2)
    images = [np.array([[1, 0],
                        [0, 0]]),
              np.array([[0, 0],
                        [0, 1]]),
              np.array([[1, 1],
                        [0, 0]]),
               np.array([[0, 1],
                         [0, 1]])         
             ]
    labels = [1, -1, 1, -1]
    print(testTraining(perceptron, images, labels))

def exercise6():
    print("\nExercise 6: Testing\n")

    perceptron = Perceptron(4,2,2)
    trainImages = [np.array([[1, 0],
                             [0, 0]]),
                   np.array([[0, 0],
                             [0, 1]]),
                   np.array([[1, 1],
                             [0, 0]]),
                   np.array([[0, 1],
                             [0, 1]])         
                  ]
    labels = [1, -1, 1, -1]
    perceptron.train(trainImages, labels, 10)

    testImages = [np.array([[1, 0],
                             [0, 0]]),
                   np.array([[0, 0],
                             [0, 1]]),
                   np.array([[1, 1],
                             [0, 0]]),
                   np.array([[0, 1],
                             [0, 1]])         
                  ]
    expectedResults = [1, -1, 1, -1]
    print(testTesting(perceptron, testImages, expectedResults))

    testImages = [np.array([[0, 0],
                            [1, 0]]),
                  np.array([[0, 0],
                            [1, 1]]),
                  np.array([[1, 1],
                            [1, 0]]),
                  np.array([[0, 1],
                            [1, 1]])         
                 ]
    expectedResults = [0, -1, 1, -1]
    print(testTesting(perceptron, testImages, expectedResults))


def bonus():
    perceptron1 = Perceptron(1,1,1)
    perceptron1.network = [Neuron((0,0), 0, False)]
    print(testSaveAndLoad(perceptron1))

    perceptron1.network = [Neuron((0,0), 0, True)]
    print(testSaveAndLoad(perceptron1))

    perceptron1.network = [Neuron((0,0), 0, True)]
    print(testSaveAndLoad(perceptron1))

    perceptron1.network = [Neuron((0,0), 1, True)]
    print(testSaveAndLoad(perceptron1))
    
    perceptron2 = Perceptron(2,2,2)
    perceptron2.network = [Neuron((0,0), -1, False), Neuron((1,1), 1, False)]
    print(testSaveAndLoad(perceptron2))

    perceptron2.network = [Neuron((0,0), -1, True), Neuron((1,1), 1, False)]
    print(testSaveAndLoad(perceptron2))

    perceptron2.network = [Neuron((0,0), -1, True), Neuron((1,1), 1, True)]
    print(testSaveAndLoad(perceptron2))

    perceptron2.network = [Neuron((0,0), -1, True), Neuron((1,1), 1, True)]
    print(testSaveAndLoad(perceptron2))

    perceptron2.network = [Neuron((0,0), -1, False), Neuron((1,1), 2, True)]
    print(testSaveAndLoad(perceptron2))

    perceptron2.network = [Neuron((0,0), 0, True), Neuron((1,1), 1, True)]
    print(testSaveAndLoad(perceptron2))

def loadImages():
    traindir = "images/train"
    testdir = "images/test"
    trainImageList = os.listdir(traindir)
    testImageList = os.listdir(testdir)
    random.shuffle(trainImageList)
    trainImages = []
    testImages = []
    labels = []
    for imageFile in trainImageList:
        im = cv2.imread(os.path.join(traindir, imageFile), cv2.IMREAD_GRAYSCALE)
        im = (im +1)%2
        trainImages.append(im)
        if imageFile[0] == "a":
            labels.append(1)
        elif imageFile[0] == "b":
            labels.append(-1)
    
    for imageFile in testImageList:
        im = cv2.imread(os.path.join(testdir, imageFile), cv2.IMREAD_GRAYSCALE)
        im = (im +1)%2
        testImages.append(im)

    return (trainImages, labels, testImages)

 # TODO Once you have completed exercises 1-6 complete the fullTest function
 # that initializes a new neural network, trains it using the trainImages and
 # labels provided then tests the results of the training with the list of test
 # images. 
def fullTest():
    trainImages, labels, testImages = loadImages()

    per = Perceptron(64, 8, 8)
    per.train(trainImages, labels, 10000)
    results = per.test(testImages)
    print(results)
   

if __name__ == "__main__":
    colorama.init()
    # exercise1()
    # exercise2()
    # exercise3()
    # exercise4()
    exercise5()
    # exercise6()
    # fullTest()
    # bonus()