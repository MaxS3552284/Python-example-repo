# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:48:36 2021

@author: Max
"""

# https://www.youtube.com/watch?v=Q-__8Xw9KTM

import pygame
import os
import random
import time
pygame.init() # initialise pygame
pygame.font.init()

# create game window:
WIDTH, HEIGHT = 750, 750
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("schpace invadorz")


# load images:

# enemy ships
# alternative: pygame.image.load('assets/pixel_ship_red_small.png') with "" or ''
RED_SPACESHIP = pygame.image.load(os.path.join('assets', 'pixel_ship_red_small.png'))
BLUE_SPACESHIP = pygame.image.load(os.path.join('assets','pixel_ship_blue_small.png'))
GREEN_SPACESHIP = pygame.image.load(os.path.join('assets', 'pixel_ship_green_small.png'))

# main ship
YELLOW_SPACESHIP = pygame.image.load(os.path.join('assets', 'pixel_ship_yellow.png'))

# bullets
RED_LASER = pygame.image.load(os.path.join('assets', 'pixel_laser_red.png'))
BLUE_LASER = pygame.image.load(os.path.join('assets','pixel_laser_blue.png'))
GREEN_LASER = pygame.image.load(os.path.join('assets', 'pixel_laser_green.png'))
YELLOW_LASER = pygame.image.load(os.path.join('assets', 'pixel_laser_yellow.png'))

# background
BG = pygame.transform.scale(pygame.image.load('assets/background-black.png'),(WIDTH, HEIGHT)) 
# background img, load and rescale to window size

class Laser:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img
        self.mask = pygame.mask.from_surface(self.img)

    def draw(self, window):
        window.blit(self.img, (self.x, self.y))

    def move(self, vel):
        self.y += vel

    def off_screen(self, height): # tells you if laser is off screen
        return not(self.y <= height and self.y >= 0)

    def collision(self, obj):
        return collide(self, obj)


class Ship:
    COOLDOWN = 30
    
    def __init__ (self, x, y, health=100):
        self.x = x
        self.y = y
        self.health = health
        self.ship_img = None
        self.laser_img = None
        self.lasers = []
        self.cool_down_counter = 0
        
    def draw(self, window):
        window.blit(self.ship_img, (self.x, self.y))
        # pygame.draw.rect(WIN, (255,0,0), (self.x, self.y, 50, 50), 0) # 0 = filled in rectangle, >0 is pen size of rectangle border
        for laser in self.lasers:
            laser.draw(window)

    def move_lasers(self, vel, obj): # if laser hit player or screen
        self.cooldown()
        for laser in self.lasers:
            laser.move(vel)
            if laser.off_screen(HEIGHT):
                self.lasers.remove(laser)
            elif laser.collision(obj):
                obj.health -= 10
                self.lasers.remove(laser)

    def cooldown(self):
        if self.cool_down_counter >= self.COOLDOWN:
            self.cool_down_counter = 0
        elif self.cool_down_counter > 0:
            self.cool_down_counter += 1

    def shoot(self):
        if self.cool_down_counter == 0: # add new laser to list when cooldown is over
            laser = Laser(self.x, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1
            
    def get_width(self):
        return self.ship_img.get_width()
    
    def get_height(self):
        return self.ship_img.get_height()
        
        
class Player(Ship):
    def __init__ (self, x, y, health=100):
        super().__init__(x, y, health)
        self.ship_img = YELLOW_SPACESHIP
        self.laser_img = YELLOW_LASER
        self.mask = pygame.mask.from_surface(self.ship_img) 
        # mask actual player area for hitbox with img area, so that we know if we hit a pixel an not just a hitbox square
        self.max_health = health
        
    def move_lasers(self, vel, objs): # if laser hit enemy or screen
        self.cooldown()
        for laser in self.lasers:
            laser.move(vel)
            if laser.off_screen(HEIGHT):
                self.lasers.remove(laser)
            else:
                for obj in objs:
                    if laser.collision(obj):
                        objs.remove(obj)
                        if laser in self.lasers:
                            self.lasers.remove(laser)
        
    def draw(self, window):
        super().draw(window)
        self.healthbar(window)
        
    def healthbar(self, window):
        pygame.draw.rect(window, (255,0,0), (self.x, self.y + self.ship_img.get_height() + 10, self.ship_img.get_width(), 10))
        pygame.draw.rect(window, (0,255,0), (self.x, self.y + self.ship_img.get_height() + 10, self.ship_img.get_width() * (self.health/self.max_health), 10))

        
class Enemy(Ship):
    COLOR_MAP = {
                "red": (RED_SPACESHIP, RED_LASER),
                "green": (GREEN_SPACESHIP, GREEN_LASER),
                "blue": (BLUE_SPACESHIP, BLUE_LASER)
                } # class variable

    def __init__(self, x, y, color, health=100):
        super().__init__(x, y, health)
        self.ship_img, self.laser_img = self.COLOR_MAP[color]
        self.mask = pygame.mask.from_surface(self.ship_img)

    def move(self, vel):
        self.y += vel

    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x-20, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1  
            
    
def collide(obj1, obj2): # tells you if two objects collide
    offset_x = obj2.x - obj1.x
    offset_y = obj2.y - obj1.y
    return obj1.mask.overlap(obj2.mask, (offset_x, offset_y)) != None


# main loop, runs game
def main():
    
    run = True
    FPS = 60 # 60 frames per second
    level = 0
    lives = 5
    main_font = pygame.font.SysFont("comicsans", 50)
    lost_font = pygame.font.SysFont("bangle", 60)
    
    enemies = []
    wave_length = 5 
    
    player_vel = 5
    enemy_vel = 1
    laser_vel = 5
    
    player=Player(300, 650)
    
    clock = pygame.time.Clock()
    
    lost = False
    lost_count = 0
    
    def redrawGameWindow(): # function inside function has access to all variables in main function
        WIN.blit(BG, (0,0))
        # lives_lable = main_font.render(f"Lives: {lives}", 1, (255,255,255))
        # level_lable = main_font.render(f"Level: {level}", 1, (255,255,255))
        lives_lable = main_font.render("Lives: " + str(lives), 1, (255,255,255)) # Arguments are: text, anti-aliasing, color
        level_lable = main_font.render("Level: " + str(level), 1, (255,255,255))
        WIN.blit(lives_lable, (10, 10))
        WIN.blit(level_lable, ((WIDTH - level_lable.get_width() - 10), 10))
        
        for enemy in enemies: # comes before draw player so if both have same position you see player
            enemy.draw(WIN)
        
        player.draw(WIN)
        
        if lost:
            lost_label = lost_font.render("You Lost!!", 1, (255,255,255))
            WIN.blit(lost_label, (WIDTH/2 - lost_label.get_width()/2, 350)) # draw it in the center of the screen
        
        pygame.display.update()
    
    
    while run:
        
        clock.tick(FPS) # runs on every computer ate the same speed since clock speed set to 60fps
        redrawGameWindow()
        
        if lives <= 0 or player.health <= 0:
            lost = True
            lost_count += 1

        if lost: # quit game 3 sec after you lost
            if lost_count > FPS * 3: # fps*3= 3sec.
                run = False
            else:
                continue

        if len(enemies) == 0:
            level += 1
            wave_length += 5
            for i in range(wave_length):
                enemy = Enemy(random.randrange(50, WIDTH-100), random.randrange(-1500, -100), random.choice(["red", "blue", "green"]))
                enemies.append(enemy)
        
        
        # check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False 
                # quit() instead of run=False will quit the entire program and not just return to main menue
                
        keys = pygame.key.get_pressed()
      
                
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and player.x - player_vel > 0: # left
            player.x -= player_vel
            
        if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and player.x + player_vel + player.get_width() < WIDTH: # right
            player.x += player_vel
            
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and player.y - player_vel > 0: # up
            player.y -= player_vel
            
        if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and player.y + player_vel + player.get_height() < HEIGHT: # down
            player.y += player_vel
            
        if keys[pygame.K_SPACE]:
            player.shoot()
            
        for enemy in enemies[:]: # enemies[:] = copy of enemies list
            enemy.move(enemy_vel)
            enemy.move_lasers(laser_vel, player)

            if random.randrange(0, 2*60) == 1: 
                # random chance per time interval(0.5sec.) for enemy to shoot
                enemy.shoot()

            if collide(enemy, player):
                player.health -= 10
                enemies.remove(enemy)
            elif enemy.y + enemy.get_height() > HEIGHT:
                lives -= 1
                enemies.remove(enemy)

        player.move_lasers(-laser_vel, enemies)
 
def main_menu():
    title_font = pygame.font.SysFont("comicsans", 70)
    run = True
    while run:
        WIN.blit(BG, (0,0))
        title_label = title_font.render("Press the mouse to begin...", 1, (255,255,255))
        WIN.blit(title_label, (WIDTH/2 - title_label.get_width()/2, 350))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                main()
    pygame.quit()        

        
        
main_menu()


































