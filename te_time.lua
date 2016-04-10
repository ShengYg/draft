#! /usr/bin/env luajit
require 'torch'
torch.setnumthreads(4)

io.stdout:setvbuf('no')   --no buffering; the result of any output operation appears immediately
for i = 1,#arg do
   io.write(arg[i] .. ' ')    -- arg 'kitty' && cmd '-a,-b...'
end
io.write('\n')
dataset = table.remove(arg, 1)

assert(dataset == 'kitti' or dataset == 'kitti2015' or dataset == 'mb')   --th main.lua kitty

cmd = torch.CmdLine()
	cmd:option('-sm_terminate', '', 'terminate the stereo method after this step')
	cmd:option('-sm_skip', '', 'which part of the stereo method to skip')

	cmd:option('-left', '')
	cmd:option('-right', '')

	cmd:option('-at', 0)
	cmd:option('-m', 0.2, 'margin')
	cmd:option('-pow', 1)

	cmd:option('-l1', 4)
	cmd:option('-fm', 64)
	cmd:option('-ks', 3)
	cmd:option('-lr', 0.002)
	cmd:option('-bs', 128)
	cmd:option('-mom', 0.9)
	cmd:option('-true1', 1)
	cmd:option('-false1', 4)
	cmd:option('-false2', 10)

	cmd:option('-L1', 0)
	cmd:option('-cbca_i1', 0)
	cmd:option('-cbca_i2', 0)
	cmd:option('-tau1', 0)
	cmd:option('-pi1', 4)
	cmd:option('-pi2', 55.72)
	cmd:option('-sgm_i', 1)
	cmd:option('-sgm_q1', 3)
	cmd:option('-sgm_q2', 2.5)
	cmd:option('-alpha1', 1.5)
	cmd:option('-tau_so', 0.02)
	cmd:option('-blur_sigma', 7.74)
	cmd:option('-blur_t', 5)

opt = cmd:parse(arg)

require 'cunn'
require 'cutorch'
require 'image'
require 'libadcensus'
require 'libcv'
require 'cudnn'

include('Normalize2.lua')
include('StereoJoin.lua')

torch.manualSeed(40)
cutorch.manualSeed(40)

function fromfile(fname)
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   if #dim == 1 and dim[1] == 0 then
      return torch.Tensor()
   end
   local x
   x = torch.FloatTensor(torch.FloatStorage(fname))
   x = x:reshape(torch.LongStorage(dim))
   return x
end

function gaussian(sigma)
   local kr = math.ceil(sigma * 3)
   local ks = kr * 2 + 1
   local k = torch.Tensor(ks, ks)
   for i = 1, ks do
      for j = 1, ks do
         local y = (i - 1) - kr
         local x = (j - 1) - kr
         k[{i,j}] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
      end
   end
   return k
end

function clean_net(net)
   net.output = torch.CudaTensor()
   net.gradInput = nil
   net.weight_v = nil
   net.bias_v = nil
   net.gradWeight = nil
   net.gradBias = nil
   net.iDesc = nil
   net.oDesc = nil
   net.finput = torch.CudaTensor()
   net.fgradInput = torch.CudaTensor()
   net.tmp_in = torch.CudaTensor()
   net.tmp_out = torch.CudaTensor()
   if net.modules then
      for _, module in ipairs(net.modules) do
         clean_net(module)
      end
   end
   return net
end

function forward_free(net, input)
   local currentOutput = input
   for i=1,#net.modules do
      local nextOutput = net.modules[i]:updateOutput(currentOutput)
      if currentOutput:storage() ~= nextOutput:storage() then
         currentOutput:storage():resize(1)
         currentOutput:resize(0)
      end
      currentOutput = nextOutput
   end
   net.output = currentOutput
   return currentOutput
end


function fix_border(net, vol, direction)          --!!
   for i=1,4 do
      vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * 5}])
   end
end

do
   local net_te = torch.load('../net/net_kitti_fast_-a_train_all.t7', 'ascii')[1]
   net_te.modules[#net_te.modules] = nil
   local X0 = fromfile('../data.kitti/x0.bin')
   local x_batch = torch.CudaTensor(2,1,350,1242)
   x_batch = X0[{{1,2}}]:cuda()
   local disp_max = 228
   t = sys.clock()
   time = {}

   local vols, vol
   forward_free(net_te, x_batch:clone())   --output(2,64,h,w)
   --net_te:forward(x_batch:clone())
   vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
   adcensus.StereoJoin(net_te.output[{{1}}], net_te.output[{{2}}], vols[{{1}}], vols[{{2}}])
   fix_border(net_te, vols[{{1}}], -1) --vols(2,disp,h,w)  vols1:output_L
   fix_border(net_te, vols[{{2}}], 1)
   clean_net(net_te)
   table.insert(time, sys.clock() - t)
   collectgarbage()
   table.insert(time, sys.clock() - t)

   disp = {}                --direction=-1:output_L
   for _, direction in ipairs({1, -1}) do
      sm_active = true

      vol = vols[{{direction == -1 and 1 or 2}}]
      sm_active = sm_active and (opt.sm_terminate ~= 'cnn')
      

      -- cross-based cost aggregation
      local x0c, x1c
      if sm_active and opt.sm_skip ~= 'cbca' then
         x0c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
         x1c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
         adcensus.cross(x_batch[1], x0c, opt.L1, opt.tau1)
         adcensus.cross(x_batch[2], x1c, opt.L1, opt.tau1)
         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
         for i = 1,opt.cbca_i1 do
            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
            vol:copy(tmp_cbca)
         end
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cbca1')
      table.insert(time, sys.clock() - t)

      --semiglobal matching
      if sm_active and opt.sm_skip ~= 'sgm' then
         vol = vol:transpose(2, 3):transpose(3, 4):clone()  --vol(1,h,w,disp)
         do
            local out = torch.CudaTensor(1, vol:size(2), vol:size(3), vol:size(4))
            local tmp = torch.CudaTensor(vol:size(3), vol:size(4))
            for _ = 1,opt.sgm_i do
               out:zero()
               adcensus.sgm2(x_batch[1], x_batch[2], vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                  opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
               vol:copy(out):div(4)
            end
            vol:resize(1, disp_max, x_batch:size(3), x_batch:size(4))
            vol:copy(out:transpose(3, 4):transpose(2, 3)):div(4)
         end
         table.insert(time, sys.clock() - t)
         collectgarbage()
         table.insert(time, sys.clock() - t)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'sgm')
      --table.insert(time, sys.clock() - t)

      if sm_active and opt.sm_skip ~= 'cbca' then
         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
         for i = 1,opt.cbca_i2 do
            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
            vol:copy(tmp_cbca)
         end
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cbca2')
      table.insert(time, sys.clock() - t)

      _, d = torch.min(vol, 2)    --dim=2, d=torch.LongTensor(1,1,h,w)
      disp[direction == 1 and 1 or 2] = d:add(-1)     --disp[1]:DR  disp[2]:DL
   end

   --disp(2,1,h,w)  dim2=(1,disp_max)
   do
      local outlier = torch.CudaTensor():resizeAs(disp[2]):zero()
      adcensus.outlier_detection(disp[2], disp[1], outlier, disp_max)
      --outlier(1,h,w)  match=0 occlusion=1 mismatch=2

      if sm_active and opt.sm_skip ~= 'occlusion' then
         disp[2] = adcensus.interpolate_occlusion(disp[2], outlier)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'occlusion')
      table.insert(time, sys.clock() - t)

      if sm_active and opt.sm_skip ~= 'mismatch' then
         disp[2] = adcensus.interpolate_mismatch(disp[2], outlier)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'mismatch')
      table.insert(time, sys.clock() - t)
   end
   if sm_active and opt.sm_skip ~= 'subpixel_enchancement' then
      disp[2] = adcensus.subpixel_enchancement(disp[2], vol, disp_max)
   end
   sm_active = sm_active and (opt.sm_terminate ~= 'subpixel_enchancement')
   table.insert(time, sys.clock() - t)

   if sm_active and opt.sm_skip ~= 'median' then
      disp[2] = adcensus.median2d(disp[2], 5)
   end
   sm_active = sm_active and (opt.sm_terminate ~= 'median')
   table.insert(time, sys.clock() - t)

   if sm_active and opt.sm_skip ~= 'bilateral' then
      disp[2] = adcensus.mean2d(disp[2], gaussian(opt.blur_sigma):cuda(), opt.blur_t)
   end
   table.insert(time, sys.clock() - t)
end

print(time)
os.exit()

