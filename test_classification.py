import numpy
from tensorflow import keras

from edgetpu.classification.engine import ClassificationEngine


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    engine = ClassificationEngine("model_edgetpu.tflite")
    for result in engine.ClassifyWithInputTensor(numpy.asarray(test_images[0]).flatten(), top_k=3):
        print("------------------")
        print(test_labels[result[0]])
        print("Score: ", result[1])


if __name__ == '__main__':
    main()
