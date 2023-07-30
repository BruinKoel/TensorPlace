import pygame

class Button:
    def __init__(self, text, pos, size, color, command):
        self.text = text
        self.pos = pos
        self.size = size
        self.color = color
        self.command = command

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos, self.size))
        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.pos[0] + self.size[0]/2, self.pos[1] + self.size[1]/2)  )
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        x, y = pos
        return self.pos[0] <= x <= self.pos[0] + self.size[0] and self.pos[1] <= y <= self.pos[1] + self.size[1]
