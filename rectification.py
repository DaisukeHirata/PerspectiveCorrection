import cv2
import numpy as np
import math
import sys

img = cv2.imread('src.jpg')
rows = img.shape[0]
cols = img.shape[1]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur  = cv2.blur(gray, (3, 3))
edges = cv2.Canny(blur, 100, 100, apertureSize = 3)



'''
Compute intersect
'''
def computeIntersect(line1, line2):
  x1, y1, x2, y2 = [ float(l) for l in line1 ]
  x3, y3, x4, y4 = [ float(l) for l in line2 ]
  v1 = np.array((x2 - x1, y2 - y1))
  v2 = np.array((x4 - x3, y4 - y3))
  dp =  np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
  if dp < 0.9:
    denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if denom != 0:
      x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
      y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    else:
      x = -1.0
      y = -1.0
  else:
    x = -1.0
    y = -1.0

  return x, y


def cornerNotExists(corners, corner):
  notExists = True
  for c in corners:
    x = c[0] - corner[0]
    y = c[1] - corner[1]
    d = math.sqrt(x*x + y*y)
    if d < 10:
      notExists = False
  return notExists


'''
Sort corners
'''
def sortCorners(corners, center):
  top = []
  bottom = []
  centerY = center[1]
  for corner in corners:
    cornerY = corner[1]
    if cornerY < centerY:
      top.append(corner)
    else:
      bottom.append(corner)

  sortedCorners = []
  if len(top) == 2 and len(bottom) == 2:
    tLeft   = top[1] if top[0][0] > top[1][0] else top[0]
    tRight  = top[0] if top[0][0] > top[1][0] else top[1]
    bLeft   = bottom[1] if bottom[0][0] > bottom[1][0] else bottom[0]
    bRight  = bottom[0] if bottom[0][0] > bottom[1][0] else bottom[1]
    sortedCorners.append(tLeft)
    sortedCorners.append(tRight)
    sortedCorners.append(bRight)
    sortedCorners.append(bLeft)

  return sortedCorners


'''
Find lines
'''
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, 30, 10)[0]


'''
Expand lines
'''
for i, (x1, y1, x2, y2) in enumerate(lines):
  x1 = float(x1)
  y1 = float(y1)
  x2 = float(x2)
  y2 = float(y2)
  v0 = 0
  if x1 - x2 != 0:
    v1 = round((y1 - y2) / (x1 - x2) * -x1 + y1)
    v2 = cols
    v3 = round((y1 - y2) / (x1 - x2) * (v2 - x2) + y2)
    lines[i] = [v0, v1, v2, v3]


'''
Find intersect
'''
corners = []
for i, line1 in enumerate(lines):
  for line2 in lines[i+1:]:
    x, y = computeIntersect(line1, line2)
    if x >= 0 and y >= 0 and cornerNotExists(corners, (x, y)):
      corners.append((x, y))


'''
check it's quadrilateral
'''
epsilon = 0.02 * cv2.arcLength(np.array(corners, np.float32), True)
approxCurve = cv2.approxPolyDP(np.array(corners, np.float32), epsilon, True)
if len(approxCurve) != 4:
  print 'The object is not quadrilateral!'
  #sys.exit(1)


'''
Get mass center
'''
accX = 0.0
accY = 0.0
for corner in corners:
  accX += corner[0]
  accY += corner[1]
m = 1.0 / len(corners)
center = (accX * m, accY * m)


'''
Sort Corners
'''
print corners
sortedCorners = sortCorners(corners, center)
sortedCorners = np.array(sortedCorners, np.float32)

'''
Draw Lines on Image
'''
for x1, y1, x2, y2 in lines:
  cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


'''
Draw Circles on Image
'''
for corner in sortedCorners:
  corner = (int(corner[0]), int(corner[1]))
  cv2.circle(img, corner, 3, (255, 0, 0), 2)

cv2.imwrite('debug.jpg', img)

'''
get transform matrix
'''
quad = np.array([[0, 0], [rows, 0], [rows, cols], [0, cols]], np.float32)
transmtx = cv2.getPerspectiveTransform(sortedCorners, quad)
dst = cv2.warpPerspective(img, transmtx, (rows, cols))
cv2.imwrite('transform.jpg', dst)


