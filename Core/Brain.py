from ArithmeticOperations.DivisionModel.Division import Division
from ArithmeticOperations.MultiplyModel.Multiply import Multiply
from ArithmeticOperations.SubstractionModel.Substraction import Substraction
from ArithmeticOperations.SumModel.Sum import Sum
from VisualOperations.HandwriteRecognition.HandwriteRecognition import HandwriteRecognition


class Brain:

    def __init__(self):
        self.division = Division()
        self.multiply = Multiply()
        self.substraction = Substraction()
        self.sum = Sum()
        self.handwrite_recognition = HandwriteRecognition()

        self.queue_operations = []


    def generate_operations():
        pass