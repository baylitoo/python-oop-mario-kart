import pygame
BLOCK_SIZE = 50
import numpy as np
from Surface import Surface

class Checkpoint(Surface):
    def __init__(self, x, y, checkpoint_id):
        # Initialize the base Surface class with the specified x and y coordinates.
        # This sets up the Checkpoint object at the given position.
        super().__init__(x, y)

        # Load the image used to visually represent the checkpoint.
        self.__finish_image = pygame.image.load("finish.png")

        # Store the unique identifier for this checkpoint.
        self.__checkpoint_id = checkpoint_id

    def draw(self, screen):
        # Draw a base rectangle at the checkpoint's position.
        pygame.draw.rect(screen, (0, 0, 0), (self._x, self._y, 50, 50))

        # Blit (copy) the checkpoint image onto the screen at the specified x and y coordinates.
        screen.blit(self.__finish_image, (self._x, self._y))
