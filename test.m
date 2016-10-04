% add path
% addpath('/home/yz/caffe-yao/matlab')

modelpath = 'models/probnet12';
model_def = fullfile(modelpath,'deploy_1.prototxt');
model_weights = fullfile(modelpath,'stage_1_final_1.caffemodel');
net = caffe.Net(model_def, model_weights, 'test');
caffe.set_mode_gpu();
caffe.set_device(0);
%%
datapath='data/zstack_1';
files=dir(fullfile(datapath,'*.tif'));
filenames=cellfun(@(x)fullfile(datapath,x),natsortfiles({files.name}),'UniformOutput',0);
num = length(filenames);
imgs = zeros(96, 96, 1, num);
for ifile = 1:num
    img = imresize(imread(filenames{ifile}),[96 96]);
    imgs(:,:,1,ifile) = single(img);
end
maximg = max(imgs(:));
minimg = min(imgs(:));
imgs = (imgs - minimg)/(maximg - minimg)*256-128;

clc
mean = [];
var = [];
for i = 1:16:201-16
    inputdata={imgs(:,:,:,i:i+15)};
    loss=net.forward(inputdata);
    mean = [mean, loss{1}];
    var = [var, loss{2}];
end

x=1:length(mean);
errorbar(x,mean,var);

% 
% for i = 1:length(net.outputs)
%     
% end