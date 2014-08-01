--------------------------------------------------------------------------------
-- Generate face, torso, person and backgroud classes
-- (for a total of 1237 bounding boxes)
--------------------------------------------------------------------------------
-- Alfredo Canziani, Jul 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'sys'
require 'image'

-- Main script------------------------------------------------------------------
extractedPath = 'Extracted-data'
if paths.dirp(extractedPath) then
   error('Destination folder already existing')
end
torsoPath = extractedPath .. '/Torso/'
facePath =  extractedPath .. '/Face/'
bodyPath =  extractedPath .. '/Body/'
bckgPath = extractedPath .. '/Bckg/'
os.execute('mkdir ' .. extractedPath)
os.execute('mkdir ' .. torsoPath)
os.execute('mkdir ' .. facePath)
os.execute('mkdir ' .. bodyPath)
os.execute('mkdir ' .. bckgPath)

for _,annotation in ipairs(sys.split(sys.ls('/Users/atcold/Work/Datasets/INRIAPerson/Train/annotations/'),'\n')) do

   -- Gets lines of bounding box coorinates
   grep = sys.execute('grep "Bounding" /Users/atcold/Work/Datasets/INRIAPerson/Train/annotations/' .. annotation)
   -- Split lines
   grepSplit = sys.split(grep,'\n')
   -- Extract box coordinates
   --[[
      Format:
      > Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (261, 109) - (511, 705)
   --]]
   boundingBox = {}
   for _,b in ipairs(grepSplit) do
      xMin, yMin, xMax, yMax = string.match(b,'(%d+), (%d+)%) %- %((%d+), (%d+)')
      table.insert(boundingBox, {
         xMin = tonumber(xMin),
         yMin = tonumber(yMin),
         xMax = tonumber(xMax),
         yMax = tonumber(yMax)
      })
      box = boundingBox[#boundingBox]
      box.x = box.xMin
      box.y = box.yMin
      box.w = box.xMax - box.xMin + 1
      box.h = box.yMax - box.yMin + 1
   end

   -- Get lines of face centre
   grep = sys.execute('grep "on object" /Users/atcold/Work/Datasets/INRIAPerson/Train/annotations/' .. annotation)
   -- Split lines
   grepSplit = sys.split(grep,'\n')
   -- Extract faces centre
   --[[
      Format:
      > Center point on object 1 "PASperson" (X, Y) : (396, 185)
   --]]
   boxCentre = {}
   for _,b in ipairs(grepSplit) do
      x, y = string.match(b,'(%d+), (%d+)')
      table.insert(boxCentre, {
         x = tonumber(x),
         y = tonumber(y)
      })
   end

   -- Load image
   --[[
      Format:
       > Image filename : "Train/pos/crop001001.png"
   --]]
   grep = sys.execute('grep "filename" /Users/atcold/Work/Datasets/INRIAPerson/Train/annotations/' .. annotation)
   fileName = string.match(grep,'"(.+)"')
   img = image.load('/Users/atcold/Work/Datasets/INRIAPerson/' .. fileName)
   fileName = string.match(fileName,'pos/(.+)%.') .. '-'

   -- Extract torso
   for nbBox, box in ipairs(boundingBox) do
      torso = img[{ {},{box.yMin,box.yMin+box.w},{box.xMin,box.xMax} }]
      --print(#torso)
      --image.display{image=image.scale(torso,256,256),legend='Torso'}
      image.savePNG(torsoPath .. fileName .. nbBox .. '.png', image.scale(torso,256,256))
   end

   -- Extract face (w/3)
   for nbBox, box in ipairs(boxCentre) do
      factor = boundingBox[nbBox].w / 3
      delta = math.floor(factor/2)
      face = img[{ {},{box.y - delta, box.y + delta},{box.x - delta,box.x + delta} }]
      --print(#face)
      --image.display{image=image.scale(face,256,256),legend='Face'}
      image.savePNG(facePath .. fileName .. nbBox .. '.png', image.scale(face,256,256))
   end

   -- Extract body
   imgW = img:size(3)
   imgH = img:size(2)
   for nbBox, box in ipairs(boundingBox) do
      bodyRect = img[{ {},{box.yMin,box.yMax},{box.xMin,box.xMax} }]
      --image.display(bodyRect)
      x0 = math.floor(box.x + box.w/2 - box.h/2)
      x1 = math.floor(box.x + box.w/2 + box.h/2 - 1)
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

      --image.display{image=image.scale(body,256,256),legend='Body'}
      image.savePNG(bodyPath .. fileName .. nbBox .. '.png', image.scale(body,256,256))
   end

   collectgarbage()

end

-- There are 1237 bounding box, 1218 bckg images
-- we need (1237 - 1218 =) 19 more bckg crops
bckgCrops = {}
for i = 1, 19 do
   idx = math.random(1218)
   bckgCrops[idx] = bckgCrops[idx] and bckgCrops[idx] + 1 or 2
end

cropWin = 128 -- images are 320 x 240
for idx, fileName in ipairs(sys.split(sys.ls('/Users/atcold/Work/Datasets/INRIAPerson/Train/neg/*'),'\n')) do
   for i = 1, bckgCrops[idx] or 1 do
      img = image.load(fileName)
      imgW = img:size(3)
      imgH = img:size(2)
      x = math.random(imgW - cropWin + 1)
      y = math.random(imgH - cropWin + 1)
      crop = img[{ {},{y,y+cropWin-1},{x,x+cropWin-1} }]

      saveName = string.match(fileName,'neg/(.+)%.')
      if bckgCrops[idx] then
         saveName = saveName .. '-' .. i
      end
      saveName = saveName .. '.png'

      --image.display{image = image.scale(crop,256,256), legend = 'Crop'}
      image.savePNG(bckgPath .. saveName, image.scale(crop,256,256))
   end
   collectgarbage()
end

