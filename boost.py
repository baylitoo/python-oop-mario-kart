import pygame
import numpy as np
BLOCK_SIZE = 50
from Surface import Surface
import math

class Boost(Surface):
    def __init__(self, x, y):
        super().__init__(x, y)  # Initialize the base Surface class with the specified x and y coordinates.

        # Load the three frames of the Boost images, named Star1.png, Star2.png, Star3.png.
        # These images will be used to create an animated effect for the Boost surface.
        self.__Boost_images = [pygame.image.load(f"Star{i}.png") for i in range(1, 4)]

        self.__road_image = pygame.image.load("road2.png")  # Load the image of the road.

        self.__anim_counter = 0  # Counter to manage the oscillation animation.
        self.__frame_switch_timer = 0  # Variable to manage frame switching for the Boost images.
        self.__current_frame = 0  # Index to track the current frame of the Boost image.

    def draw(self, screen):
        # Calculate oscillation for vertical movement.
        # The sine function creates a smooth oscillation, and dividing the anim_counter by 10 slows down the oscillation.
        # Multiplying by 5 increases the range of movement, making the oscillation more noticeable.
        oscillation = 5 * math.sin(self.__anim_counter / 10)

        # Update the y-coordinate by adding the oscillation to create a floating effect.
        animated_y = self._y + oscillation

        # Switch to the next Boost image frame every 8 frames.
        # Reset the frame_switch_timer after each cycle and update the current_frame index.
        if self.__frame_switch_timer >= 8:
            self.__frame_switch_timer = 0  # Reset the frame switch timer.
            # Cycle through the Boost image frames.
            self.__current_frame = (self.__current_frame + 1) % len(self.__Boost_images)

        # Draw a rectangle at the Boost's position.
        pygame.draw.rect(screen, (0, 0, 0), (self._x, self._y, 50, 50))

        # Blit the road image at the Boost's position.
        screen.blit(self.__road_image, (self._x, self._y))
        # Blit the current Boost image at the updated y-coordinate (animated_y) to create the floating effect.
        screen.blit(self.__Boost_images[self.__current_frame], (self._x, animated_y))

        # Increment the animation counter and frame switch timer.
        self.__anim_counter += 1
        self.__frame_switch_timer += 1





