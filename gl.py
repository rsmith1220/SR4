#Rebecca Smith
#Seccion 20

import struct
from collections import namedtuple
import random
import numpy as np
from obj import Obj
from math import cos, sin, pi

v2 = namedtuple('Point2', ['x', 'y'])
v3 = namedtuple('Point3', ['x', 'y', 'z'])
v4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])

def char(c):
    #1 byte
    return struct.pack('=c', c.encode('ascii'))

def word(w):
    #2 bytes
    return struct.pack('=h', w)

def dword(d):
    #4 bytes
    return struct.pack('=l', d)

def color(r, g, b):
    return bytes([int(b * 255),
                  int(g * 255),
                  int(r * 255)] )
                

def baryCoords(A, B, C, P):
        areaPBC = (B.y - C.y) * (P.x - C.x) + (C.x - B.x) * (P.y - C.y)
        areaPAC = (C.y - A.y) * (P.x - C.x) + (A.x - C.x) * (P.y - C.y)
        areaABC = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)
        try:
            # PBC / ABC
            u = areaPBC / areaABC
            # PAC / ABC
            v = areaPAC / areaABC
            # 1 - u - v
            w = 1 - u - v
        except:
            return -1, -1, -1
        else:
            return u, v, w

class Renderer(object):
    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.clearColor = color(0,0,0)
        self.currColor = color(1,1,1)
        self.active_shader = None

        self.dirLight = v3(1,0,0)

        self.glViewport(0,0,self.width, self.height)
        
        self.glClear()
    
    def glViewport(self, posX, posY, width, height):
        self.vpX = posX
        self.vpY = posY
        self.vpWidth = width
        self.vpHeight = height

    def glClearColor(self, r, g, b):
        self.clearColor = color(r,g,b)

    def glColor(self, r, g, b):
        self.currColor = color(r,g,b)

    def glClear(self):
        self.pixels = [[ self.clearColor for y in range(self.height)]
                       for x in range(self.width)]
        self.zbuffer = [[ float('inf') for y in range(self.height)]
                          for x in range(self.width)]

    def glClearViewport(self, clr = None):
        for x in range(self.vpX, self.vpX + self.vpWidth):
            for y in range(self.vpY, self.vpY + self.vpHeight):
                self.glPoint(x,y,clr)


    def glPoint(self, x, y, clr = None): # Window Coordinates
        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[x][y] = clr or self.currColor

    def glPoint_vp(self, ndcX, ndcY, clr = None): # NDC
        if ndcX < -1 or ndcX > 1 or ndcY < -1 or ndcY > 1:
            return

        x = (ndcX + 1) * (self.vpWidth / 2) + self.vpX
        y = (ndcY + 1) * (self.vpHeight / 2) + self.vpY

        x = int(x)
        y = int(y)

        self.glPoint(x,y,clr)


    def glCreateRotationMatrix(self, pitch = 0, yaw = 0, roll = 0):
        
        pitch *= pi/180
        yaw   *= pi/180
        roll  *= pi/180

        pitchMat = [[1, 0, 0, 0],
                              [0, cos(pitch),-sin(pitch), 0],
                              [0, sin(pitch), cos(pitch), 0],
                              [0, 0, 0, 1]]

        yawMat = [[cos(yaw), 0, sin(yaw), 0],
                            [0, 1, 0, 0],
                            [-sin(yaw), 0, cos(yaw), 0],
                            [0, 0, 0, 1]]

        rollMat = [[cos(roll),-sin(roll), 0, 0],
                             [sin(roll), cos(roll), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]

        #multiplicacion de matrices
        result11 = [[sum(pitchMat * yawMat for pitchMat, yawMat in zip(tr_row, rot_row))
                        for rot_row in zip(*yawMat)]
                                for tr_row in pitchMat]
 
        result22 = [[sum(result11 * rollMat for result11, rollMat in zip(r1_row, roll_row))
                        for roll_row in zip(*rollMat)]
                                for r1_row in result11]

        return result22
        
    def glCreateObjectMatrix(self, translate = v3(0,0,0), rotate = v3(0,0,0), scale = v3(1,1,1)):

        translation = [[1, 0, 0, translate.x],
                        [0, 1, 0, translate.y],
                        [0, 0, 1, translate.z],
                        [0, 0, 0, 1]]

        rotation = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]

        scaleMat = [[scale.x, 0, 0, 0],
                    [0, scale.y, 0, 0],
                    [0, 0, scale.z, 0],
                    [0, 0, 0, 1]]

        #multiplicacion de matrices
        result1 = [[sum(translation * rotation for translation, rotation in zip(tr_row, rot_row))
                        for rot_row in zip(*rotation)]
                                for tr_row in translation]
 
        result2 = [[sum(result1 * scaleMat for result1, scaleMat in zip(r1_row, scale_row))
                        for scale_row in zip(*scaleMat)]
                                for r1_row in result1]


 
 
        return result2





    def glLoadModel(self, filename, translate = v3(0,0,0), rotate = v3(0,0,0), scale = v3(1,1,1)):
        model = Obj(filename)
        modelMatrix = self.glCreateObjectMatrix(translate, rotate, scale)
        rotate = v3(0, 180, 0), 
        for face in model.faces:
            vertCount = len(face)

            for vert in range(vertCount):

                index0 = face[vert][0] - 1
                index1 = face[(vert + 1) % vertCount ][0] - 1
                
                 
                
                vert0 = model.vertices[index0]
                vert1 = model.vertices[index1]
                
                x0 = round( (vert0[0]*scale.x)+ translate.x)
                y0 = round( (vert0[1]*scale.y)+ translate.y)
                
                x1 = round( (vert1[0]*scale.x)+ translate.x)
                y1 = round( (vert1[1]*scale.y)+ translate.y)
                
            
                
                self.glLine(v2(x0,y0), v2(x1, y1))

        if vertCount == 4:
            b3 = model.vertices[ face[3][0] - 1]
            b3 = self.glTransform(b3, modelMatrix)
            

       
    def glTransform(self, vertex, matrix):
    
        v = v4(vertex[0], vertex[1], vertex[2], 1)
        vt = matrix @ v
        vt = vt.tolist()[0]
        vf = v3(vt[0] / vt[3],
                vt[1] / vt[3],
                vt[2] / vt[3])

        return vf

    


    def glTriangle_bc(self, A, B, C, clr = None):
        # bounding box
        minX = round(min(A.x, B.x, C.x))
        minY = round(min(A.y, B.y, C.y))
        maxX = round(max(A.x, B.x, C.x))
        maxY = round(max(A.y, B.y, C.y))

        triangleNormal = np.cross( np.subtract(B, A), np.subtract(C,A))
        # normalizar
        triangleNormal = triangleNormal / np.linalg.norm(triangleNormal)


        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                def glTriangle_bc(self, A, B, C, clr = None):

                    if z < self.zbuffer[x][y]:
                        self.zbuffer[x][y] = z
                        self.glPoint(x,y, clr)

                        if self.active_shader:
                            r, g, b = self.active_shader(self,
                            baryCoords=(u,v,w),
                            vColor = clr or self.currColor,
                            triangleNormal = triangleNormal)



                            self.glPoint(x, y, color(r,g,b))
                        else:
                            self.glPoint(x,y, clr)



                



    def glLine(self, v0, v1, clr = None):
        # Bresenham line algorithm
        # y = m * x + b
        x0 = int(v0.x)
        x1 = int(v1.x)
        y0 = int(v0.y)
        y1 = int(v1.y)

        # Si el punto0 es igual al punto 1, dibujar solamente un punto
        if x0 == x1 and y0 == y1:
            self.glPoint(x0,y0,clr)
            return

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        # Si la linea tiene pendiente mayor a 1 o menor a -1
        # intercambio las x por las y, y se dibuja la linea
        # de manera vertical
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        # Si el punto inicial X es mayor que el punto final X,
        # intercambio los puntos para siempre dibujar de 
        # izquierda a derecha       
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        offset = 0
        limit = 0.5
        m = dy / dx
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                # Dibujar de manera vertical
                self.glPoint(y, x, clr)
            else:
                # Dibujar de manera horizontal
                self.glPoint(x, y, clr)

            offset += m

            if offset >= limit:
                if y0 < y1:
                    y += 1
                else:
                    y -= 1
                
                limit += 1
    

    

       







    def glFinish(self, filename):
        with open(filename, "wb") as file:
            # Header
            file.write(bytes('B'.encode('ascii')))
            file.write(bytes('M'.encode('ascii')))
            file.write(dword(14 + 40 + (self.width * self.height * 3)))
            file.write(dword(0))
            file.write(dword(14 + 40))

            #InfoHeader
            file.write(dword(40))
            file.write(dword(self.width))
            file.write(dword(self.height))
            file.write(word(1))
            file.write(word(24))
            file.write(dword(0))
            file.write(dword(self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))

            #Color table
            for y in range(self.height):
                for x in range(self.width):
                    file.write(self.pixels[x][y])





