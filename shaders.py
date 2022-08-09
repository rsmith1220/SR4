import numpy as np

def flat(render, **kwargs):

    u, v, w = kwargs["baryCoords"]
    b, g, r = kwargs["vColor"]
    triangleNormal = kwargs["triangleNormal"]

    b /= 255
    g /= 255
    r /= 255

    dirLight = np.array(render.dirLight)
    intensity = np.dot(triangleNormal, -dirLight)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0