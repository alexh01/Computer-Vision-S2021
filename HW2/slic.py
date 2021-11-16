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
    ###this method finds the lowest gradient possible for the pixel in this image
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
    # here I am able to sort the list of tuples by the final element in ascending order and take the first entry
    tup = sorted(gradPos, key=lambda e: e[2])[0]
    x,y = tup[0],tup[1]
    return (x,y,img1[x][y][0],img1[x][y][1],img1[x][y][2])# I return the x,y coordinates and rgb values of the found pixel

# here I use the rgb gradients to get the initial cluster centers as requested
img0 = np.array(rgb_im)
CC_Is = []
# using these loops I start at position 25,25 and get the initial cluster center for every 50x50 segment
for x in range(25, width, 50):
    for y in range(25, height, 50):
        CC_Is.append((x,y,img0[x][y][0],img0[x][y][1],img0[x][y][2]))
CC_Is = [lowestGradPos(img0,c[0],c[1]) for c in CC_Is]
# here I create a grid the size of the image to store distances of each pixel to a given cluster center
# I set the highest distances to infinity and fill my array with that
di = np.ndarray((width,height))
highestPosDist = np.inf
for row in range(width):
    for col in range(height):
        di[row][col] = highestPosDist

# this is a simular array for the labels/clusters each pixel belongs to
# each pixel is initally at the cluster dubbed -1 (so None)
li = np.zeros((width,height))
for row in range(width):
    for col in range(height):
        li[row][col] = -1

###This method calculates the residual error by using the euclidean distance with respect to x and y between 2 centers
def residualError(oldc, newc):
    re = 0
    for a,b in zip(oldc,newc):
        for index in range(0,2):
            re +=abs(a[index]-b[index])
    return re/300

###This recursive function employs the bulk of my SLIC algo
def assignment(oldcenters, img1, iters, dis, lis):
    ###in this code block I define and initialize a list containing each cluster as a list of pixels
    ###I also update my arrays dis and lis
    clusters = []
    for index in range(len(oldcenters)):
        clusters.append([])
    num = 0
    for center in oldcenters:
        for xpos in range(max(0,int(center[0]-50)),min(int(center[0]+50),int(width))):
            for ypos in range(max(0,int(center[1]-50)),min(int(center[1]+50),int(height))):
                pix = (xpos,ypos,img1[xpos][ypos][0],img1[xpos][ypos][1],img1[xpos][ypos][2])
                D = euclideanDist(center,pix)
                if(D<=dis[xpos][ypos]):
                    dis[xpos][ypos] = D
                    lis[xpos][ypos] = num
                clusters[int(lis[xpos][ypos])].append(pix)
        num += 1
    num = 0
            
            
    print('iteration %d'%iters)
    ###In this code block I fill the new centers with the avg values of the pixels in its corresponding cluster
    newcenters = [(0,0,0,0,0)]*len(oldcenters)
    for cluster in clusters:
        for pixel in cluster:
            newcenters[num] = tuple([a+b/len(cluster) for a,b in zip(newcenters[num],pixel)])
        num += 1
    # I set err = to the residual error using my old and new cluster centers
    err = abs(residualError(oldcenters,newcenters))
    print(err)
    # Here I see if I have met the threshold/exceeded the limit of iterations
    # if not I recursively call this function, otherwise I return the centers clusters and labels
    if err > 0.01 and iters < 10:
        return assignment(newcenters, img1, iters+1, dis, lis)
    return (newcenters,clusters,lis)

# I call and store the results of my SLIC algo function
tup = assignment(CC_Is, img0, 0, di, li)
cs = tup[0]
cl = tup[1]
li = tup[2]

####this code block gets the mean rgb values of each cluster storing them in t
def mean(clustr):
    m = [0,0,0]
    for i in range(2,5):
        for ele in clustr:
            m[i-2] += ele[i]
        m[i-2] = m[i-2]//len(clustr)
    return m

# same block as in K-means
t = []
for n in cl:
    t.append(mean(n))

####this code block will create a new image where the rgb values of each cluster are set to the cluster's avg
imf = Image.open("wt_slic.png")
vals = imf.load()

# same as in last q
distance = []
num = 0
for l in cl:
    for ele in l:
        vals[ele[1],ele[0]] = (t[num][0],t[num][1],t[num][2])
    num+=1

# here is where I check if each pixel is an edge pixel or not by comparing it's label to it's adjacent pixels' labels
def check(grid, x, y):
    retval = 0
    if(x>0 and x<len(grid)-1):
        if(y>0 and y<len(grid[0])-1):
            for i in range(-1,2):
                for j in range(-1,2):
                    if not (grid[x+i][y+j] == grid[x][y]):
                        return True
            return False
    return True

print(li)

# I save what I had before outlining the area around clusters
imf.save("slic_segmentation2.png")

# Here I set the pixels that are on the edges of clusters to black using my function above
for xp in range(len(li)):
    for yp in range(len(li[xp])):
        if(check(li,xp,yp)):
            vals[yp,xp] = (0,0,0)

# finally I save my result
imf.save("slic_segmentation.png")
