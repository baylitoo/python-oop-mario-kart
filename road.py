import pygame
import numpy as np
from Surface import Surface

BLOCK_SIZE = 50
class Road(Surface):  # Road class inherits from Surface, making it a type of Surface.
    def __init__(self, x, y):
        # Initialize the base Surface class with the specified x and y coordinates.
        # This sets up the Road object at the given position.
        super().__init__(x, y)

        # Load the road image. This image will be used to visually represent the road on the screen.
        self.__road_image = pygame.image.load("road2.png")

    def draw(self, screen):
        # Draw a base rectangle at the road's position.
        pygame.draw.rect(screen, (0, 0, 0), (self._x, self._y, 50, 50))

        # Blit (copy) the road image onto the screen at the specified x and y coordinates.
        screen.blit(self.__road_image, (self._x, self._y))
