--------------------------------------------------------------------------------
-- Check the position of a bounding box
--------------------------------------------------------------------------------
-- Alfredo Canziani, Jul 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'image'
require 'qtwidget'

-- Main script------------------------------------------------------------------
-- Load image
img = image.load('/Users/atcold/Work/Datasets/INRIAPerson/Train/pos/crop001001.png')

-- Annotation format
--[[
Original label for object 1 "PASperson" : "UprightPerson"
Center point on object 1 "PASperson" (X, Y) : (396, 185)
Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (261, 109) - (511, 705)
--]]

-- Create list box centres
boxCentre = {}
table.insert(boxCentre, {x = 396, y = 185})
table.insert(boxCentre, {x = 119, y = 385})
table.insert(boxCentre, {x = 219, y = 235})

-- Create list box coordinates
boundingBox = {}
table.insert(boundingBox, {xMin = 261, yMin = 109, xMax = 511, yMax = 705})
table.insert(boundingBox, {xMin =  31, yMin = 326, xMax = 209, yMax = 712})
table.insert(boundingBox, {xMin = 148, yMin = 179, xMax = 290, yMax = 641})

for _,b in ipairs(boundingBox) do
   b.x = b.xMin
   b.y = b.yMin
   b.w = b.xMax - b.xMin + 1
   b.h = b.yMax - b.yMin + 1
end

-- Display image and get handle
win = qtwidget.newwindow(img:size(3), img:size(2))
image.display{image = img, win = win}

-- Display box centres
win:setcolor(1,0,0)
win:arc(boxCentre[1].x, boxCentre[1].y, 5, 0, 360)
win:fill()
win:rectangle(boundingBox[1].x, boundingBox[1].y, boundingBox[1].w, boundingBox[1].h)
win:stroke()

win:setcolor(0,1,0)
win:arc(boxCentre[2].x, boxCentre[2].y, 5, 0, 360)
win:fill()
win:rectangle(boundingBox[2].x, boundingBox[2].y, boundingBox[2].w, boundingBox[2].h)
win:stroke()

win:setcolor(0,0,1)
win:arc(boxCentre[3].x, boxCentre[3].y, 5, 0, 360)
win:fill()
win:rectangle(boundingBox[3].x, boundingBox[3].y, boundingBox[3].w, boundingBox[3].h)
win:stroke()

-- Extract torso
for nbBox, box in ipairs(boundingBox) do
   torso = img[{ {},{box.yMin,box.yMin+box.w},{box.xMin,box.xMax} }]
   --print(#torso)
   image.display{image=image.scale(torso,256,256),legend='Torso'}
end

-- Extract face (w/3)
for nbBox, box in ipairs(boxCentre) do
   factor = boundingBox[nbBox].w / 3
   delta = math.floor(factor/2)
   face = img[{ {},{box.y - delta, box.y + delta},{box.x - delta,box.x + delta} }]
   --print(#face)
   image.display{image=image.scale(face,256,256),legend='Face'}
end

-- Extract body
imgW = img:size(3)
imgH = img:size(2)
for nbBox, box in ipairs(boundingBox) do
   bodyRect = img[{ {},{box.yMin,box.yMax},{box.xMin,box.xMax} }]
   --image.display(bodyRect)
   x0 = math.floor(box.x + box.w/2 - box.h/2)
   x1 = math.floor(box.x + box.w/2 + box.h/2) - 1
   x0crop = x0 > 0 and x0 or 1
   x1crop = x1 > imgW and imgW or x1

   bodyCrop = img[{ {},{box.yMin,box.yMax},{x0crop,x1crop} }]
   --image.display(bodyCrop)

   body = torch.Tensor(3,box.h,box.h)
   if x0 < 0 then
      x0 = x0 - 1
      for i = 1,-x0 do
         body[{ {},{},i }] = img[{ {},{box.yMin,box.yMax},1 }]
      end
   else x0 = 0 end
   --print(#bodyCrop,x0)
   body[{ {},{},{1-x0,bodyCrop:size(3)-x0} }] = bodyCrop
   if x1 > imgW then
      for i = box.h-(x1-imgW),box.h do
         body[{ {},{},i }] = img[{ {},{box.yMin,box.yMax},imgW }]
      end
   end

   image.display{image=image.scale(body,256,256),legend='Body'}
end
