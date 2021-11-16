# Problem 2: SLIC. (65 points) Apply a variant of the SLIC algorithm to wt_slic.png, by implementing
# the following steps:
# 1. Divide the image in blocks of 50x50 pixels and initialize a centroid at the center of each
# block.
# 2. Compute the magnitude of the gradient in each of the RGB channels and use the square root
# of the sum of squares of the three magnitudes as the combined gradient magnitude. Move
# the centroids to the position with the smallest gradient magnitude in 3x3 windows centered
# on the initial centroids.
# 3. Apply k-means in the 5D space of x,y,R,G,B. Use the Euclidean distance in this space, 
# but divide x and y by 2.
# 4. After convergence, display the output image as in slide 41 of week 6: color pixels that touch
# two different clusters black and the remaining pixels by the average RGB value of their
# cluster.

from PIL import Image, ImageDraw
import numpy as np
import random
import math

im = Image.open("wt_slic.png")
rgb_im = im.convert('RGB')

height,width = im.size

def euclideanDist(pointA, pointB):
    #for each dimenstion(xyRGB) in A and B square the difference
    #finally take the square root of the sum of those values and return it
    d = 0.0
    for index in range(5):
        if(index<=1):
            d += (0.5*float(pointA[index])-0.5*float(pointB[index]))**2
        else:
            d += (float(pointA[index])-float(pointB[index]))**2
    d = math.sqrt(d)
    return d

def lowestGradPos(img1,xval,yval):
    gradPos = []
    for i in range(xval-1,xval+2):
        for j in range(yval-1,yval+2):#so this is the 3x3 window.
            #magnitude of gradient in R
            mgR = (int(img1[i+1][j][0])-int(img1[i-1][j][0]))**2 + (int(img1[i][j+1][0])-int(img1[i][j-1][0]))**2
            #magnitude of gradient in G
            mgG = (int(img1[i+1][j][1])-int(img1[i-1][j][1]))**2 + (int(img1[i][j+1][1])-int(img1[i][j-1][1]))**2
            #magnitude of gradient in B
            mgB = (int(img1[i+1][j][2])-int(img1[i-1][j][2]))**2 + (int(img1[i][j+1][2])-int(img1[i][j-1][2]))**2
            gradPos.append((i,j,math.sqrt(mgR+mgG+mgB)))
    tup = sorted(gradPos, key=lambda e: e[2])[0]
    x,y = tup[0],tup[1]
    return (x,y,img1[x][y][0],img1[x][y][1],img1[x][y][2])

img0 = np.array(rgb_im)
CC_Is = []
for x in range(25, width, 50):
    for y in range(25, height, 50):
        try:
            CC_Is.append((x,y,img0[x][y][0],img0[x][y][1],img0[x][y][2]))
        except IndexError:
            print(x)
            print(y)
            exit()


CC_Is = [lowestGradPos(img0,c[0],c[1]) for c in CC_Is]
di = np.ndarray((width,height))
highestPosDist = math.sqrt(0.25*(width**2+height**2)+(255**2)*3)
for row in range(width):
    for col in range(height):
        di[row][col] = highestPosDist

def residualError(oldc, newc):
    re = 0
    for a,b in zip(oldc,newc):
        for index in range(len(a)):
            re += (a[index]-b[index])/len(a)#5
    return re/len(oldc)#k

def assignment(oldcenters, img1, k, dis):
    clusters = []
    for index in range(k):
        clusters.append([])
    num = 0
    for center in oldcenters:
        for xpos in range(max(0,center[0]-50),min(center[0]+50,width)):
            for ypos in range(max(0,center[1]-50),min(center[1]+50,height)):
                pix = (xpos,ypos,img1[xpos][ypos][0],img1[xpos][ypos][1],img1[xpos][ypos][2])
                D = euclideanDist(center,pix)
                if(D<dis[xpos][ypos]):
                    dis[xpos][ypos] = D
                    clusters[num].append(pix)
        num += 1
    converged = False
    num = 0
    newcenters = [(0,0,0,0,0)]*k
    for cluster in clusters:
        for pixel in cluster:
            newcenters[num] = tuple([a+b/len(cluster) for a,b in zip(newcenters[num],pixel)])
        num += 1
    if residualError(oldcenters,newcenters) > 5*k:
        return assignment(newcenters, img1, k, dis)
    return (newcenters,clusters)


tup = assignment(CC_Is, img0, len(CC_Is), di)
cs = tup[0]
cl = tup[1]
print(cs)

####this code block gets the mean rgb values of each cluster storing them in t
def mean(clustr):
    m = [0,0,0]
    for i in range(2,5):
        for ele in clustr:
            m[i-2] += ele[i]
        m[i-2] = m[i-2]//len(clustr)
    return m


t = []
for n in range(len(cl)):
    t.append(mean(cl[n]))


####this code block will create a new image where the rgb values of each cluster are set to the cluster's avg
imf = Image.open("wt_slic.png")
vals = imf.load()

distance = []
for xn in range(height):
    for yn in range(width):
        try:
            vals[xn,yn] = (0,0,0)
        except IndexError:
            print(xn)
            print(yn)
            exit()
o, num = width*height,0
for l in cl:
    for ele in l:
        vals[ele[1],ele[0]] = (t[num][0],t[num][1],t[num][2])
        if(o%100==0):
            print("%d"%o)
        o -= 1
    num+=1
imf.save("slic_segmentation.png")