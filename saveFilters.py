# saveFilters.m python versioin
#caffe.reset_all();
#clear; close all;
#%% settings
import caffe
caffe.set_mode_cpu()
folder = './'
#folder = '../'
model = folder + 'SRCNN_mat.prototxt'
weights = folder + 'SRCNN_iter_500.caffemodel'
savepath = folder + 'x3.mat'
layers = 3

#%% load model using mat_caffe
net = caffe.Net(model,weights,caffe.TEST);

#%% reshap parameters
weights_conv = [];


#for layer_name, param in net.params:
#    print (layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

for idx in range(1, layers+1):
    #conv_filters = net.layers(['conv' + str(idx)]).param[1].get_data();
    
    conv_filters = net.layers('conv1').params(1).get_data();
    
    tmp,fsize,channel,fnum= conv_filters.size();
    
    if channel == 1 :
        weights = single(ones(fsize^2, fnum));
    else:
        weights = single(ones(channel, fsize^2, fnum));
    
    for i in (1,channel+1):
        for j in (1,fnum+1):
             temp = conv_filters[:,:,i,j];
             if channel == 1:
                weights[:,j] = temp[:];
             else:
                weights[i,:,j] = temp[:];

    weights_conv[idx] = weights;

#%% save parameters
weights_conv1 = weights_conv[1];
weights_conv2 = weights_conv[2];
weights_conv3 = weights_conv[3];
biases_conv1 = net.layers('conv1').params(2).get_data();
biases_conv2 = net.layers('conv2').params(2).get_data();
biases_conv3 = net.layers('conv3').params(2).get_data();

save(savepath,'weights_conv1','biases_conv1','weights_conv2','biases_conv2','weights_conv3','biases_conv3');

