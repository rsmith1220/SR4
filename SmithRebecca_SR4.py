#Rebecca Smith
#Seccion 20

from gl import Renderer, color, V2, V3
from textures import Texture
from shaders import flat, gourad
from obj import Obj
import random

from shaders import flat


w=600
h=600
z=10

rend= Renderer(w,h)
rend.active_shader = flat
rend.active_texture = Texture("body.bmp")



rend.glLoadModel("kirby.obj",
                translate = V3(w/2, h/2, z/2),
                rotate = V3(0, 180, 0),
                scale = V3(300,300,300)
                 )

rend.glFinish("output.bmp")

