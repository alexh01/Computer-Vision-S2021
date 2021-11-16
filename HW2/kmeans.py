# Apply k-means segmentation on white-tower.png with k=10. 
# The distance function should only consider the RGB color channels and ignore pixel coordinates.
# Randomly pick 10 RGB triplets from the existing pixels as initial seeds and run to convergence.
# After k-means has converged, represent each cluster with the average RGB value of its members,
# creating an image as in slide 18 of Week 6.
from PIL import Image, ImageDraw
import numpy as np
import random
import math#imported libraries
# 1. Randomly initialize the cluster centers, c_1, ..., c_K
# 2. Given cluster centers, determine points in each cluster
# - For each point p, find the closest c_i. Put p into cluster i
# 3. Given points in each cluster, solve for c_i
# - Set c_i to be the mean of points in cluster i
# 4. If c_i have changed, repeat Step 2
im = Image.open("white-tower.png")
rgb_im = im.convert('RGB')#given data

def euclideanDist(pointA, pointB):
    # for each dimenstion(RGB) in A and B square the difference
    # finally take the square root of the sum of those values and return it
    d = 0.0
    for index in range(3):
        d += (float(pointA[index])-float(pointB[index]))**2
    d = math.sqrt(d)
    return d

def checker(listA, listB):
    # in this method I iterate through the elements of each list
    # if any are not equal I return False, otherwise I return True
    for i in range(len(listA)):
        if (listA[i] != listB[i]):
            return False
    return True

def kmeans(img0, k):
    # here the pix list contains the tuples containing each pixel's rgb values
    pix = []
    # I iterate through the given image (converted to a 2darray) and add the elements(RGB-values) to pix
    for row in np.array(img0):
        for cell in row:
            pix.append(list(cell))
    # here the cluster centers are randomly initialized from 10 random points
    centers = random.sample(pix, k)
    print(centers)
    # here I define clusters
    clusters = [[]]*k
    print(clusters)
    # this keeps track of the distance everything is from a given pixel
    dist = []
    # the index used later
    i = 0
    # these values are used between iterations in my loop
    avgR = 0
    avgG = 0
    avgB = 0
    # used to break out of the loop
    converged = False
    while(not converged):
        # here I set the old centers equal to our current ones
        oldcenters = []
        for center in centers:
            oldcenters.append(center.copy())
        ####this code block fills each cluster with the pixels closest to the working center
        # clear for our new clusters
        clusters = [[],[],[],[],[],[],[],[],[],[]]
        # iterate through pixels
        for pixel in pix:
            # clear dist
            dist = []
            # fill the dist with each point's distance from each cluster center
            for center in centers:
                dist.append(euclideanDist(pixel,center))
            # fill the clusters with the k pixels with the least distances from its center
            for index in range(k):
                if dist[index] == min(dist):
                    clusters[index].append(pixel)
                    break
        # print results of iteration
        for cluster in clusters:
            print(len(cluster),end=' ')
        print()
        print(centers)
        ####this code block updates the centers
        # here the mean rgb values of each cluster are found and set as the new centers
        i = 0
        for cluster in clusters:
            avgR = 0
            avgG = 0
            avgB = 0
            for point in cluster:
                avgR += point[0]
                avgG += point[1]
                avgB += point[2]
            centers[i][0] = avgR//len(cluster)
            centers[i][1] = avgG//len(cluster)
            centers[i][2] = avgB//len(cluster)
            i += 1
        # here we check if our new centers equal our old ones if so we exit the loop else we repeat
        converged = checker(oldcenters,centers)
    return (centers,clusters)

# calls my k-means function storing the clusters and centers
tup = kmeans(rgb_im,10)
cs = tup[0]
cl = tup[1]
print(cs)

####this code block gets the mean rgb values of each cluster storing them in t
def mean(clustr):
    m = [0,0,0]
    for i in range(3):
        for ele in clustr:
            m[i] += ele[i]
        m[i] = m[i]//len(clustr)
    return m
# calls mean with my clusters to get the rgb-values to be set in our new image
t = []
for n in range(10):
    t.append(mean(cl[n]))


####this code block will create a new image where the rgb values of each cluster are set to the cluster's avg
imf = Image.open("white-tower.png")
vals = imf.load()
# the dimensions of the image are stored here
width, height = im.size
distance = []
for x in range(width):
    for y in range(height):
        # here we get the distances between each center and our current pixel storing them in dist
        distance = []
        for c in cs:
            distance.append(euclideanDist(vals[x,y],c))
        # finally we find the closest cluster center
        # use that to determine the value in t containing the color
        # and set the pixel's color in our new image to that color in t
        for index in range(10):
                if distance[index] == min(distance):
                    vals[x,y] = (t[index][0],t[index][1],t[index][2])
                    break
        if(x%100==0 and y%100==0):
            print("%d,%d"%(x,y))
# here we save the image and are finished
imf.save("segmentation.png")
#Image.close(img1)