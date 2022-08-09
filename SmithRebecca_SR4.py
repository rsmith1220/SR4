#Rebecca Smith
#Seccion 20

from gl import Renderer, color, v2, v3
from obj import Obj
import random

from shaders import flat


w=1080
h=1080
z=10

rend= Renderer(w,h)
rend.active_shader = flat


rend.glLoadModel("girl.obj",
                translate = v3(w/2, h/2, z/2),
                rotate = v3(0, 180, 0),
                scale = v3(300,300,300)
                 )

rend.glFinish("output.bmp")