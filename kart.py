import numpy as np
import pygame
from Physics import Physics
import time
from pygame import mixer
import math

CELL_SIZE = 50


# Kart class that manages the update of its position, drawing, and other actions.
# It inherits several methods and parameters from Physics class.
class Kart(Physics):
    MAX_ACCELERATION = 0.25   # Maximum acceleration
    MAX_ANGLE_VELOCITY = 0.05  # Maximum angular velocity

    def __init__(self, controller):
        super().__init__()
        mixer.init()
        self.__countdown_played = False   # Countdown counter
        # Load the kart image
        self.__Kart_image = pygame.image.load("Kart4.png")  # Kart image
        self.__start_time = time.time()  # Initialize start time
        self.controller = controller   # Initialize controller
        self.controller.kart = self    # Initialize Kart instance in the controller
        self.__ac_f = 0                  # Forward acceleration
        self.__ac_b = 0                  # Backward acceleration
        self.__angle_velocity_l = 0      # Left angular velocity
        self.__angle_velocity_r = 0      # Right angular velocity
        self.__current_surface = None    # Current surface of the kart
        self.__checkpoints_positions = {} # Checkpoints positions
        self.__TOTAL_CHECKPOINTS = 0      # Total amount of checkpoints
        self.has_finished = False
        self.__lava_entry_time = None     # Setting the time when the kart gets into Lava or out of track timer
        self.__checkpoint_id_map = {'C': 0, 'D': 1, 'E': 2, 'F': 3}   # Dictionary with all the possible checkpoint characters and id order
        self.__countdown_sound = mixer.Sound("countdown-sound-effect-8-bit-151797.mp3") #play countdown sound



    # Current surface getter
    @property
    def current_surface(self):
        return self.__current_surface



    # Reset function that resets the kart position, velocity, and other parameters
    def reset(self, initial_position, initial_orientation):
        self.__start_time = time.time()  # Record the current time
        self._x, self._y = initial_position
        self._xc, self._yc = 50, 50
        self._θc = 0
        self._angle = initial_orientation
        self._velocity = 0
        self._next_checkpoint_id = 0
        self.has_finished = False

    # Forward Method
    def forward(self):
        self.__ac_f = self.MAX_ACCELERATION

    # Backward Method
    def backward(self):
        self.__ac_b = self.MAX_ACCELERATION

    # Turn Left Method
    def turn_left(self):
        self.__angle_velocity_l = self.MAX_ANGLE_VELOCITY

    # Turn Right Method
    def turn_right(self):
        self.__angle_velocity_r = self.MAX_ANGLE_VELOCITY

    # Method that helps us get the surface type of the current surface that the kart is on
    def __get_surface_type(self, track_string):
        grid_x = int(self._x // CELL_SIZE)   # Getting the line position of the current surface
        grid_y = int(self._y // CELL_SIZE)   # Getting the column position of the current surface
        track_width = len(track_string.split('\n')[0])  # Getting the track width using one line of the track string array
        track_height = len(track_string.split('\n'))    # Getting the track height using track string array

        # In case the kart gets into a surface outside the track (it should be impossible since we respawn the kart to the last xc yx position)
        if grid_y < 0 or grid_y >= track_height or grid_x < 0 or grid_x >= track_width:
            return 'Unknown'

        return track_string.split('\n')[grid_y][grid_x]  # Returning the character of the surface [C, E, D, F, G, R, L, B]



    # Method that helps us get the current state of the kart for the IA Input
    def get_state(self, track_string):
        self.__calculate_checkpoints_positions(track_string)   # Getting all the checkpoint positions

        # Getting the closest checkpoint position as well as the distance
        closest_checkpoint_x, closest_checkpoint_y, distance_to_checkpoint = self.__find_closest_checkpoint(
            self._next_checkpoint_id)

        # Getting the angle between the checkpoint and our kart
        angle_to_checkpoint = self.__calculate_angle_to_checkpoint(closest_checkpoint_x, closest_checkpoint_y)

        surrouding_surfaces= self.__get_surrounding_surfaces(track_string) # Get the surrounding surfaces type

        return np.array([distance_to_checkpoint,angle_to_checkpoint, self._velocity] ), np.array(surrouding_surfaces)
        # Return the current state of the kart


    # Methode that helps us get the sourrounding surfaces types of the kart for the AI
    def __get_surrounding_surfaces(self, track_string):
        # Taille de la grille à examiner autour du kart
        grid_size = 1  # Vous pouvez ajuster cela selon le besoin

        # Convertir la chaîne de caractères de la piste en grille
        track_grid = track_string.split('\n')
        grid_x = int(self._x // CELL_SIZE)
        grid_y = int(self._y // CELL_SIZE)

        # Directions à examiner : devant, derrière, gauche, droite
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]  # Get all 8 surfaces surrounding the kart
        surrounding_surfaces = []

        for dx, dy in directions:
            check_x, check_y = grid_x + dx, grid_y + dy

            # Vérifier si la position est dans les limites de la piste
            if 0 <= check_x < len(track_grid[0]) and 0 <= check_y < len(track_grid):
                surface = track_grid[check_y][check_x]
            else:
                surface = 'Unknown'  # Hors piste ou en dehors des limites

            surrounding_surfaces.append(surface)

        return surrounding_surfaces


    # Method that returns the closest checkpoint position and distance
    def __find_closest_checkpoint(self, checkpoint_id):
        closest_distance = float('inf')

        # Iterating all positions of the next checkpoint
        for x, y in self.__checkpoints_positions[checkpoint_id]:
            distance = self.__calculate_distance_to_checkpoint(x, y) # Calculating the distance

            # Saving the distance and the x, y values of the nearest checkpoint block
            if distance < closest_distance:
                closest_distance = distance
                x_c, y_c = x, y
        return x_c, y_c, closest_distance



    # parcours de la chaine de caractère pour trouver le prochain checkpoint , nous pouvons faire mieux exemple recherche dichotomique
    # qui commence par la position actuelle du kart
    def __calculate_checkpoints_positions(self, track_string):
        checkpoint_id_map = {'C': 0, 'D': 1, 'E': 2, 'F': 3}
        self.__checkpoints_positions = {id: [] for id in checkpoint_id_map.values()}  # Dictionary with the id of each checkpoint
        #that should save the positions of each checkpoint block inside


        track_lines = track_string.split('\n')
        #iterating all the line off the track string
        for y, line in enumerate(track_lines):
            #iterating all checkpoints ids and characters
            for checkpoint_char, checkpoint_id in checkpoint_id_map.items():
                position = 0
                while position != -1:
                    position = line.find(checkpoint_char, position)
                    if position != -1:
                        self.__checkpoints_positions[checkpoint_id].append((position * CELL_SIZE, y * CELL_SIZE))
                        position += 1  # Continue la recherche après la position actuelle


    # Method to calculate the distance between two positions
    def __calculate_distance_to_checkpoint(self, checkpoint_x, checkpoint_y):
        return np.sqrt((checkpoint_x - self._x) ** 2 + (checkpoint_y - self._y) ** 2)

    # Method to calculate the angle between two positions
    def __calculate_angle_to_checkpoint(self, checkpoint_x, checkpoint_y):
        # Angle from current position to checkpoint
        return np.arctan2(checkpoint_y - self._y, checkpoint_x - self._x) - self._angle




    # Method that updates the positions of the kart based on the surface it is on and the commands F, B, L, R
    def update_position(self, track_string, screen):

        current_time = time.time()
        if current_time - self.__start_time <= 3:  # 3 seconds duration

            pass

        else:

            # Find the highest order of checkpoint character present in the track_string
            self.__TOTAL_CHECKPOINTS = max((self.__checkpoint_id_map[char] for char in track_string if char in self.__checkpoint_id_map),
                                       default=-1)

            # Getting the current surface that the kart is on
            self.__current_surface = self.__get_surface_type(track_string)

            # Checking if the kart is off the limits of the track
            self._Off_limits(track_string)

            # Checks if the Kart is out of limits
            if self._Off_L==True:
                self._Off_L=False  # Reset counter

            else:
                # Calculating the current acceleration and angle velocity
                self._ac = self.__ac_f - self.__ac_b    #Aditionning the Forward and Backward accelerations for correct implementation
                self._angle_velocity = -self.__angle_velocity_l + self.__angle_velocity_r  #Aditionning the Left and Right Velocity angles for correct implementation


                # Conditions that help activate the effects of each surface whether or not the kart is on that surface
                if self.__current_surface == "G":  # Grass
                    self._physic_grass()
                elif self.__current_surface == "R":  # Road
                    self._physic_road()
                elif self.__current_surface == "L":  # Lava
                    self._physic_lava()
                elif self.__current_surface == "B":  # Boost
                    self._physic_boost()
                elif self.__current_surface in ['C', 'D', 'E', 'F']:  # Checkpoint
                    self._physic_checkpoint(self.__current_surface,self.__TOTAL_CHECKPOINTS)



            # Reset of these values after each update so that the physics are correct
            self.__ac_f = 0
            self.__ac_b = 0
            self.__angle_velocity_l = 0
            self.__angle_velocity_r = 0
            self._ac=0
            self._angle_velocity=0


            # Reset the kart, used for IA training
            # self.new_state=self.get_state(track_string)
            # if self.controller.name=="AI":
            #     if time.time() - self.__start_time >= 120:
            #         initial_position = [50, 50]
            #         initial_angle = 0
            #         self.controller.train()
            #         self.reset(initial_position, initial_angle)






    # Draw Method for kart with Respawning animation
    def draw(self, screen):

        #Activates the Starting the race sound at the start of the game
        if not self.__countdown_played:
            current_time = time.time()
            if current_time - self.__start_time <= 3:  # 3 seconds duration
                self.__countdown_sound.play()
                self.__countdown_played = True  # Sets the variable to true so that we don't play the sound another time




        # Check if the kart is respawning due to lava
        if self._got_into_lava_or_Off_limits:
            current_time = time.time()

            # Start the timer when the kart first enters the lava
            if self.__lava_entry_time is None:
                self.__lava_entry_time = current_time

            elapsed_time = current_time - self.__lava_entry_time

            # Create a blinking effect for 1.5 seconds
            if elapsed_time <= 1.5:
                # Blink every 0.1 second. If the current part of the time interval is even, draw the kart.
                if int(elapsed_time * 10) % 2 == 0:  # Integer part of elapsed time * 10 is used for blinking control
                    rotated_kart_image = pygame.transform.rotate(self.__Kart_image, -math.degrees(self._angle))
                    screen.blit(rotated_kart_image, (self._x - 33, self._y - 31))  #25 22
            else:
                # After 1.5 seconds, stop the blinking effect and reset the flags
                self._got_into_lava_or_Off_limits = False
                self.__lava_entry_time = None
        else:
            # Draw the kart normally if it's not in lava
            rotated_kart_image = pygame.transform.rotate(self.__Kart_image, -math.degrees(self._angle))
            screen.blit(rotated_kart_image, (self._x - 33, self._y - 31))



