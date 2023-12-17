import pygame
import numpy as np
BLOCK_SIZE = 50

import pygame
from Surface import Surface


class Lava(Surface):
    def __init__(self, x, y):
        # Initialize the base Surface class with the specified x and y coordinates.
        super().__init__(x, y)

        # Load the five frames of the lava animation.
        # These images will be used to create an animated effect for the lava surface.
        self.__lava_images = [pygame.image.load(f"lava_frame_{i}.png") for i in range(1, 6)]

        # A counter to manage the timing of animation frame changes.
        self.__animation_counter = 1

    def draw(self, screen):
        # Draw a base rectangle at the lava's position.
        pygame.draw.rect(screen, (0, 0, 0), (self._x, self._y, 50, 50))

        # Determine which frame of the lava animation to display.
        # The animation counter is divided by 4 (integer division) to slow down the frame switch rate.
        # This determines the current index in the lava_images list.
        image_index = (self.__animation_counter - 1) // 4

        # Draw the current frame of the lava animation at the specified position.
        screen.blit(self.__lava_images[image_index], (self._x, self._y))

        # Increment the animation counter and reset it when it reaches 20.
        # This cycle creates a continuous animation loop for the lava.
        self.__animation_counter = 1 if self.__animation_counter == 20 else self.__animation_counter + 1

