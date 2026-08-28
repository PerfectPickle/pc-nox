from abc import ABC, abstractmethod

class ModelBase(ABC):
    """
    Interface for all Model classses.
    """

    # @abstract_method
    # def build_activities():
    #     """
    #     Builds or rebuilds activities (latent states).
    #     """
    #     pass

    @abstractmethod
    def predict():
        """
        Runs every layer's `predict` function once, given required args.
        Used by energy function.
        """
        pass