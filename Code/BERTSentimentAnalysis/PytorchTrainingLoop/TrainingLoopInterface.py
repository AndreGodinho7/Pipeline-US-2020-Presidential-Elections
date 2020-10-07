from abc import ABC, abstractmethod

class TrainingLoopInterface(ABC):
    @abstractmethod
    def trainloop(self, classifier, train, val, epochs, classifier_name, **kwargs):
        pass