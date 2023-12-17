import pygame
from Physics import Physics
import numpy as np
BLOCK_SIZE = 50
from Surface import Surface
class Grass(Surface):
    def __init__(self, x, y):
        # Initialize the base Surface class with the specified x and y coordinates.
        # This sets up the Grass object at the given position.
        super().__init__(x, y)

        # Load the grass image. This image will be used to represent the grass surface on the screen.
        self.__grass_image = pygame.image.load("grass2.png")

    def draw(self, screen):
        # Draw a rectangle at the grass's position.
        pygame.draw.rect(screen, (0, 0, 0), (self._x, self._y, 50, 50))

        # Blit (copy) the grass image onto the screen at the specified x and y coordinates.
        screen.blit(self.__grass_image, (self._x, self._y))


