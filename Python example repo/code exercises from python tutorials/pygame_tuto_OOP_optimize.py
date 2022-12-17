# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:15:46 2021

@author: Max
"""

# pygame playlist
# https://www.youtube.com/playlist?list=PLzMcBGfZo4-lp3jAExUCewBfMx3UZFkh5

# video 4+:

import pygame
pygame.init() # initialise pygame

# create window to draw in (hier: 500x500 pixel groß)
win = pygame.display.set_mode((500,480))
# caption/name of window
pygame.display.set_caption("first game")

bulletSound = pygame.mixer.Sound("Game/Game_bullet.mp3")
hitSound = pygame.mixer.Sound("Game/Game_hit.mp3")

music = pygame.mixer.music.load("Game/music.mp3")
pygame.mixer.music.play(-1) # -1 will ensure the song keeps looping

screenwith = 500
clock = pygame.time.Clock()
score = 0

# This goes outside the while loop, near the top of the program
# lists containing the animation images, when move later, the list contents will be displayed one after another
walkRight = [pygame.image.load('Game/R1.png'), pygame.image.load('Game/R2.png'), pygame.image.load('Game/R3.png'), pygame.image.load('Game/R4.png'), pygame.image.load('Game/R5.png'), pygame.image.load('Game/R6.png'), pygame.image.load('Game/R7.png'), pygame.image.load('Game/R8.png'), pygame.image.load('Game/R9.png')]
walkLeft = [pygame.image.load('Game/L1.png'), pygame.image.load('Game/L2.png'), pygame.image.load('Game/L3.png'), pygame.image.load('Game/L4.png'), pygame.image.load('Game/L5.png'), pygame.image.load('Game/L6.png'), pygame.image.load('Game/L7.png'), pygame.image.load('Game/L8.png'), pygame.image.load('Game/L9.png')]
bg = pygame.image.load('Game/bg.jpg') # background img
char = pygame.image.load('Game/standing.png')



class player(object):
    
    def __init__(self, x, y, width, height):
        # character atributes 
        self.x = x # x start coord
        self.y = y # y start coord
        self.width = width # character size, size of character image files
        self.height = height
        self.velocity = 5 # stepsize
        self.isJump = False
        self.left = False # to keep track in which direction you move
        self.right = False # to keep track in which direction you move
        self.walkCount = 0 # counts number of steps
        self.jumpCount = 10
        self.standing = True
        self.hitbox = (self.x + 17, self.y + 11, 29, 52)

    def draw(self, win):

        # create red rectangle as character
        # old:
        # pygame.draw.rect(win, (255,0,0), (x, y, width, height)) # (255,0,0) = color in rbg
        # new:
        if self.walkCount + 1 >= 27:
            # 27 since 9 animation frames and each is displayed for 3 sec.
            self.walkCount = 0
        
        if not (self.standing): 
            if self.left:  
                win.blit(walkLeft[self.walkCount//3], (self.x, self.y))
                # // = integer division, dicards remainder: bsp: 4//3=1
                self.walkCount += 1                          
            elif self.right:
                win.blit(walkRight[self.walkCount//3], (self.x, self.y))
                self.walkCount += 1
        else:
            # win.blit(char, (self.x, self.y)) # old:
            # self.walkCount = 0
            if self.right: # new: video 5
                win.blit(walkRight[0], (self.x, self.y)) # show first image of animation
            else:
                win.blit(walkLeft[0], (self.x, self.y))
        self.hitbox = (self.x + 17, self.y + 11, 29, 52)
        # pygame.draw.rect(win, (255,0,0), self.hitbox,2) # To draw the hit box around the player
        
    def hit(self):
        self.isJump = False # resett jump if you got hit during jump, otherwise you land under the screen frame
        self.jumpCount = 10
        self.x = 60 # We are resetting the player position
        self.y = 410
        self.walkCount = 0
        font1 = pygame.font.SysFont('bangle', 100)
        text = font1.render('-5', 1, (255,0,0))
        win.blit(text, (250 - (text.get_width()/2),200))
        pygame.display.update()
        i = 0
        while i < 300:
            pygame.time.delay(10)
            i += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    i = 301
                    pygame.quit()
            
            # After we are hit we are going to display a message to the screen for
            # a certain period of time


class projectile(object):
    def __init__(self, x, y, radius, color, facing):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.facing = facing # gonna be either 1 or -1, the direction the character faces
        self.velocity = 8 * facing

    def draw(self,win):
        pygame.draw.circle(win, self.color, (self.x,self.y), self.radius)


class enemy(object):
    walkRight = [pygame.image.load('Game/R1E.png'), pygame.image.load('Game/R2E.png'), pygame.image.load('Game/R3E.png'), pygame.image.load('Game/R4E.png'), pygame.image.load('Game/R5E.png'), pygame.image.load('Game/R6E.png'), pygame.image.load('Game/R7E.png'), pygame.image.load('Game/R8E.png'), pygame.image.load('Game/R9E.png'), pygame.image.load('Game/R10E.png'), pygame.image.load('Game/R11E.png')]
    walkLeft = [pygame.image.load('Game/L1E.png'), pygame.image.load('Game/L2E.png'), pygame.image.load('Game/L3E.png'), pygame.image.load('Game/L4E.png'), pygame.image.load('Game/L5E.png'), pygame.image.load('Game/L6E.png'), pygame.image.load('Game/L7E.png'), pygame.image.load('Game/L8E.png'), pygame.image.load('Game/L9E.png'), pygame.image.load('Game/L10E.png'), pygame.image.load('Game/L11E.png')]
    
    def __init__(self, x, y, width, height, end):
        
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.path = [x, end]
        self.walkCount = 0
        self.velocity = 3
        self.hitbox = (self.x + 17, self.y + 2, 31, 57)
        self.health = 10
        self.visible = True

    def draw(self, win):
        self.move()
        
        if self.visible:
            if self.walkCount + 1 >= 33:
                self.walkCount = 0
            
            if self.velocity > 0:
                win.blit(self.walkRight[self.walkCount//3], (self.x,self.y))
                self.walkCount += 1
            else:
                win.blit(self.walkLeft[self.walkCount//3], (self.x,self.y))
                self.walkCount += 1
            
            # red (255,0,0) || darker green (0,128,0)
            pygame.draw.rect(win, (255,0,0), (self.hitbox[0], self.hitbox[1] - 20, 50, 10))
            pygame.draw.rect(win, (0,128,0), (self.hitbox[0], self.hitbox[1] - 20, 50 - (5 * (10 - self.health)), 10))

            self.hitbox = (self.x + 17, self.y + 2, 31, 57) # 
            # pygame.draw.rect(win, (255,0,0), self.hitbox,2) # Draws the hit box around the enemy
            
    def move(self):
        if self.velocity > 0:
            if self.x < self.path[1] + self.velocity:
                self.x += self.velocity
            else:
                self.velocity = self.velocity * -1
                self.x += self.velocity
                self.walkCount = 0
        else:
            if self.x > self.path[0] - self.velocity:
                self.x += self.velocity
            else:
                self.velocity = self.velocity * -1
                self.x += self.velocity
                self.walkCount = 0
    
    def hit(self):  # This will display when the enemy is hit
        hitSound.play() 
        if self.health > 0:
            self.health -= 1
        else:
            self.visible = False
        print('hit')

    
def redrawGameWindow():
    win.blit(bg, (0,0))
    text = font.render("Score: " + str(score), 1, (0,0,0)) # Arguments are: text, anti-aliasing, color
    win.blit(text, (340, 10))
    man.draw(win)
    goblin.draw(win)
    for bullet in bullets:
        bullet.draw(win)
    
    pygame.display.update()



font = pygame.font.SysFont("bangle", 30, True)
# The first argument is the font, next is size 
# and then True to make our font bold

man = player(200, 410, 64,64)
goblin = enemy(100, 410, 64, 64, 300)
shootLoop = 0
bullets = []
# mainloop of game, always need a mainloop to check on collision etc.
# running main loop = game running
run = True
while run:
    # pygame.time.delay(50)# 50ms|| old: 100ms
    clock.tick(27) # 27sec?
    
    if goblin.visible == True:
        if man.hitbox[1] < goblin.hitbox[1] + goblin.hitbox[3] and man.hitbox[1] + man.hitbox[3] > goblin.hitbox[1]:
            if man.hitbox[0] + man.hitbox[2] > goblin.hitbox[0] and man.hitbox[0] < goblin.hitbox[0] + goblin.hitbox[2]:
                man.hit()
                score -= 5
    
    # basic cooldown timer for shooting
    if shootLoop > 0: 
        shootLoop += 1
    if shootLoop > 3: # when 4 is reached, restett to 0 and cooldown is over
        shootLoop = 0


    # check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    for bullet in bullets:
        # check if bullets are in goblin hitbox
        if goblin.visible == True:
            if bullet.y - bullet.radius < goblin.hitbox[1] + goblin.hitbox[3] and bullet.y + bullet.radius > goblin.hitbox[1]:
                if bullet.x + bullet.radius > goblin.hitbox[0] and bullet.x - bullet.radius < goblin.hitbox[0] + goblin.hitbox[2]:
                    goblin.hit()
                    score += 1
                    bullets.pop(bullets.index(bullet)) # delete bullet when goblin is hit
                
        if bullet.x < 500 and bullet.x > 0:
            bullet.x += bullet.velocity # bullet moves in whatever direction its shot
        else:
            bullets.pop(bullets.index(bullet)) #.pop = remove element || here, remove bullet if screen border is reached 
            
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_SPACE] and shootLoop == 0:
        bulletSound.play()
        if man.left:
            facing = -1 # if we move left we move in negative direction since upper left corner of sceen in coord (0,0)
        else:
            facing = 1
            
        if len(bullets) < 5: # max. 5 bullets on the screen at once
            # round(man.x + man.width //2), so that the bullets come from the middle of the man, 
            # and decimal numbers would mess with the object on the screen
            # (0,0,0) = black
            bullets.append(projectile(round(man.x + man.width //2), round(man.y + man.height//2), 6, (0,0,0), facing))
        
        shootLoop = 1
    
    if keys[pygame.K_LEFT] and man.x > man.velocity: # conditions after "and" for movement limited window
        man.x -= man.velocity
        man.left = True
        man.right = False
        man.standing = False
    elif keys[pygame.K_RIGHT] and man.x < (screenwith-man.width-man.velocity):
        man.x += man.velocity
        man.left = False
        man.right = True
        man.standing = False
    else:
        # man.left = False 
        # man.right = False
        man.standing = True
        man.walkCount = 0
        
    if not (man.isJump): # cant move up or down while junping
        # if keys[pygame.K_UP] and man.y > man.velocity:
        #     man.y -= man.velocity
        # if keys[pygame.K_DOWN] and man.y < (screenwith-man.height-man.velocity):
        #     man.y += man.velocity
        if keys[pygame.K_UP]:
            man.isJump = True
            man.left = False 
            man.right = False
            man.walkCount = 0
    else:
        if man.jumpCount >= -10:
            neg = 1 # to keep target cooords of jump negative, for landing so to not "fly away"
            
            if man.jumpCount < 0:
                neg = -1
                
            man.y -= (man.jumpCount ** 2) * 0.5 * neg # **2 = squared
            man.jumpCount -= 1     
        else:
            man.isJump = False
            man.jumpCount = 10
            
        
    redrawGameWindow()
    
    
pygame.quit()








































