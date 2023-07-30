import  Models.AudioRNN as audiornn
import Datasets.WAV.WavSet

import pygame
from Gui.Button import Button
import Models.AudioRNN as audiornn
import Datasets.WAV.WavSet
from Host.AudioHost import AudioHost

model = None
audiohost = None

def load_model():
    global  audiohost
    audiohost.load_model()
    # Your code to load the .pth model file goes here
    pass

def train_model():
    global  audiohost
    audiohost.train(dataset_path = "DATA/WAV")
    # Your code to train the model goes here
    pass

def save_model():
    global  audiohost
    audiohost.save_model()
    # Your code to save the model goes here
    pass



def new_model():
    global  model, audiohost
    audiohost = AudioHost()

    model = audiohost.model




def main():
    global  model
    new_model()
    pygame.init()

    # Set up the screen
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Audio Model Interface")

    # Create buttons
    load_model_button = Button("Load Model", (50, 50), (150, 50), (0, 255, 0), command=load_model)
    train_model_button = Button("Train Model", (50, 150), (150, 50), (0, 0, 255), command=train_model)
    save_model_button = Button("Save Model", (50, 250), (150, 50), (255, 0, 0), command=save_model)
    laod_dataset_button = Button("Load Dataset", (50, 350), (150, 50), (255, 0, 0), command=Datasets.WAV.WavSet.WavSet)

    buttons = [load_model_button, train_model_button, save_model_button]

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    if button.is_clicked(mouse_pos):
                        button.command()

        # Clear the screen
        screen.fill((255, 255, 255))

        # Draw the buttons
        for button in buttons:
            button.draw(screen)

        # Update the screen
        pygame.display.flip()

    # Quit pygame
    pygame.quit()
