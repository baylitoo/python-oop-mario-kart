from abc import ABC, abstractmethod

class Surface(ABC):
    # Abstract class that serves as a base for different surface types.
    # It provides a structure that includes the 'draw' method, which must be implemented by all subclasses.
    def __init__(self, x, y):
        # Initialize the Surface with x and y coordinates.
        self._x = x
        self._y = y

    @abstractmethod
    def draw(self, screen):
        # Abstract method that must be implemented by all subclasses.
        # This method is responsible for drawing the surface on the screen.
        pass
