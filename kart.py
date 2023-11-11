import pygame

MAX_ANGLE_VELOCITY = 0.05
MAX_ACCELERATION = 0.25


class Kart():  # Vous pouvez ajouter des classes parentes
    """
    Classe implementant l'affichage et la physique du kart dans le jeu
    """
    
    def __init__(self, controller):
        self.has_finished = False
        self.controller = controller
        # A modifier et completer
        pass
       
    def reset(self, initial_position, initial_orientation):
        # A completer
        pass
        
    def forward(self):
        # A completer
        pass
    
    def backward(self):
        # A completer
        pass
    
    def turn_left(self):
        # A completer
        pass
        
    def turn_right(self):
        # A completer
        pass
    
    def update_position(self, string, screen):
        # A completer
        pass
    
    def draw(self, screen):
        # A modifier et completer
        kart_position = [75, 75]
        kart_radius = 20
        
        # Draw a circle
        pygame.draw.circle(screen, (255, 255, 255), kart_position, kart_radius)

    # Completer avec d'autres methodes si besoin (ce sera probablement le cas)