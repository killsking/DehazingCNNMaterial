# DehazingCNNMaterial
He Zhang (S’14) received the Ph.D. degree in
electrical and computer engineering from Rutgers
University, New Brunswick, NJ, USA, in 2018.
He is currently a Research Scientist with Adobe,
San Jose, CA, USA. His research interests include
image restoration, image compositing, generative
adversarial network, deep learning, and sparse and
low-rank representation.



from collections import namedtuple
from cv2.ximgproc import guidedFilter
from net import *
from net.losses import StdLoss
from net.vae_model import VAE
from utils.imresize import np_imresize
from utils.image_io import *
from utils.dcp import get_atmosphere
from fusion import ImageDehazing
from skimage.io import imread, imsave
#from skimage.measure import compare_psnr, compare_ssim
import torch
import torch.nn as nn
import numpy as np
from model import GridDehazeNet
DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a'])
def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return gradient_h, gradient_y

class Dehaze(object):
    def __init__(self, image_name, image, num_iter=1000, clip=True, output_path="output/"):
        self.image_name = image_name
        self.image = image
        self.num_iter = num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.003
        self.parameters = None
        self.current_result = None
        self.output_path = output_path

        self.clip = clip
        self.blur_loss = None
        self.best_result_psnr = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 3
        self.post = None
        self.adaptiveavg= nn.AdaptiveAvgPool2d(1)
        self.batch_size = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.images_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.image_net  = GridDehazeNet(height=3, width=6, num_dense_layer=4, growth_rate=16).type(data_type)
        self.image_net = nn.DataParallel(self.image_net, device_ids=[0])
        self.image_net.load_state_dict(torch.load('{}_haze_best_{}_{}'.format('outdoor', 3, 6)))
        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)

    def _init_ambient(self):
        ambient_net = VAE(self.image.shape)
        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)

        atmosphere = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)
        self.at_back = atmosphere

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        self.image_net_inputs = np_to_torch(self.image).cuda()
        self.mask_net_inputs = np_to_torch(self.image).cuda()
        self.ambient_net_input = np_to_torch(self.image).cuda()
        self.gradie_h_gt, self.gradie_v_gt=gradient(self.image_net_inputs)

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        self.batch_size = self.images_torch.shape
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure()
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self):
        self.image_out = self.image_net(self.image_net_inputs)
        self.ambient_out = self.ambient_net(self.ambient_net_input)

        self.mask_out = self.mask_net(self.mask_net_inputs)
        #self.mask_out[self.mask_out<0.05]=0.05
        self.blur_out = self.blur_loss(self.mask_out)
        gradie_h_gt, gradie_v_gt=gradient(self.image_net_inputs)
        
        gradie_h_est, gradie_v_est=gradient(self.image_out)
        
        self.gradie_h_gt=torch.max(gradie_h_est,gradie_h_gt).detach()
        self.gradie_v_gt=torch.max(gradie_v_est,gradie_v_gt).detach()
        L_tran_v = self.mse_loss(gradie_v_est, self.gradie_v_gt)+self.mse_loss(gradie_h_est, self.gradie_h_gt)

        #self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * tmpair,
                                     #self.images_torch)+
        tmpair=self.adaptiveavg((1 - self.mask_out)*self.images_torch)/self.adaptiveavg(1 - self.mask_out)+0.1
        tmpair[tmpair<0.6]=0.6
        tmpimg=(self.images_torch-tmpair)/self.mask_out+tmpair
        smooth_x,smooth_y=gradient(self.mask_out)

            
        #print (batch_size)
        #tmpambient_out=tmpair.repeat(1,1,self.batch_size[2],self.batch_size[3])
        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                     self.images_torch)+(torch.mean(smooth_x)+torch.mean(smooth_x))*0.05+L_tran_v*0.5+(torch.mean(gradie_h_est)+torch.mean(gradie_v_est))#+self.mse_loss(self.image_out,tmpimg)
        vae_loss = self.ambient_net.getLoss()
        self.total_loss = self.mseloss + vae_loss
        self.total_loss += 0.005 * self.blur_out
        self.total_loss +=self.mse_loss(torch.mean(self.image_out),torch.mean(self.image_net_inputs))
        dcp_prior = torch.min(self.image_out.permute(0, 2, 3, 1), 3)[0]
        self.dcp_loss = self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.05
        self.total_loss += self.dcp_loss
        if (self.current_result):
            self.total_loss += self.mse_loss(self.ambient_out, np_to_torch(self.current_result.a).type(torch.cuda.FloatTensor))
            self.total_loss += self.mse_loss(self.mask_out, np_to_torch(self.current_result.t).type(torch.cuda.FloatTensor))
            self.total_loss += self.mse_loss(self.image_out, np_to_torch(self.current_result.learned).type(torch.cuda.FloatTensor))
        #self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        self.total_loss += self.mse_loss(self.ambient_out, tmpair * torch.ones_like(self.ambient_out))

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        if step % 5 == 0:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
            ambient_out_np[0,:,:]=self.A_matting(ambient_out_np[0,:,:])
            ambient_out_np[1,:,:]=self.A_matting(ambient_out_np[1,:,:])
            ambient_out_np[2,:,:]=self.A_matting(ambient_out_np[2,:,:])
            mask_out_np[mask_out_np<0.1]=0.1
            mask_out_np = self.t_matting(mask_out_np)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np)

    def _plot_closure(self, step):
        """
         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f  %f\n' % (step, self.total_loss.item(), self.blur_out.item()), '\r', end='')

    def finalize(self):
        self.final_t_map = np_imresize(self.current_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.current_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        mask_out_np[mask_out_np<0.1]=0.1
        self.final_a[0,:,:]=self.A_matting(self.final_a[0,:,:])
        self.final_a[1,:,:]=self.A_matting(self.final_a[1,:,:])
        self.final_a[2,:,:]=self.A_matting(self.final_a[2,:,:])
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        dehazer = ImageDehazing(verbose=False)
        #print (self.original_image.shape,post.shape)
       
        posttest = dehazer.dehaze([np_to_pil(self.original_image),np_to_pil(post)], pyramid_height=15)
        posttest['dehazed'] = np.clip(posttest['dehazed'], 0, 1)
        #posttest['dehazed'].save(output_path + "{}.jpg".format(self.image_name))
        imsave('output/'+self.image_name+'.jpg', posttest['dehazed'])
        #save_image(self.image_name + "_run_final", posttest, self.output_path)
        save_image(self.image_name + "_run_final_out", self.current_result.learned, self.output_path)

    def A_matting(self, mask_out_np):
        #print (self.original_image.transpose(1, 2, 0).shape,mask_out_np.astype(np.float32).shape)
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np.astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])
    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(image_name, image, num_iter=350, output_path="output/"):
    dh = Dehaze(image_name, image, num_iter, clip=True, output_path=output_path)

    dh.optimize()
    dh.finalize()

    save_image(image_name + "_original", np.clip(image, 0, 1), dh.output_path)

if __name__ == "__main__":
    torch.cuda.set_device(0)

    hazy_add = 'data/canon_input.png'
    name = "canon_input111"
    print(name)

    hazy_img = prepare_hazy_image(hazy_add)
    #print (hazy_img.shape)
    dehaze(name, hazy_img, num_iter=500, output_path="output/")




import numpy as np

import cv2 as cv

# skimage imports
from skimage.util import img_as_ubyte, img_as_float64
from skimage.color import rgb2gray
from skimage.color import rgb2hsv

class ImageDehazing:
    def __init__(self, verbose=False):
        '''Function to initialize class variables'''
        self.image = None
        self.verbose = verbose

    def __clip(self, image=None):
        '''Function to clip images to range of [0.0, 1.0]'''
        # Validate parameters
        if image is None:
            return None

        image[image < 0] = 0
        image[image > 1] = 1
        return image

    def __show(self, images=None, titles=None, size=None, gray=False):
        '''Function to display images'''
        # Validate parameters
        if images is None or titles is None or size is None:
            return

        plt.figure(figsize=size)

        plt.subplot(1, 2, 1)
        if gray is True:
            plt.imshow(images[0], cmap='gray')
        else:
            plt.imshow(images[0])
        plt.title(titles[0])
        plt.axis('off')

        plt.subplot(1, 2, 2)
        if gray is True:
            plt.imshow(images[1], cmap='gray')
        else:
            plt.imshow(images[1])
        plt.title(titles[1])
        plt.axis('off')

        plt.show()

    def white_balance(self, image=None):
        '''Function to perform white balancing operation on an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)

        # Extract colour channels
        R = image[:, :, 2]
        G = image[:, :, 1]
        B = image[:, :, 0]

        # Obtain average intensity for each colour channel
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        mean_RGB = np.array([mean_R, mean_G, mean_B])

        # Obtain scaling factor
        grayscale = np.mean(mean_RGB)
        scale = grayscale / mean_RGB

        white_balanced = np.zeros(image.shape)

        # Rescale original intensities
        white_balanced[:, :, 2] = scale[0] * R
        white_balanced[:, :, 1] = scale[1] * G
        white_balanced[:, :, 0] = scale[2] * B

        # Clip to [0.0, 1.0]
        white_balanced = self.__clip(white_balanced)

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
                images=[self.image, white_balanced],
                titles=['Original Image', 'White Balanced Image'],
                size=(15, 15)
            )
        return white_balanced

    def enhance_contrast(self, image=None):
        '''Function to enhance contrast in an image'''
        # Validate parameters
        if image is None:
            return None

        image = img_as_float64(image)

        # Extract colour channels
        R = image[:, :, 2]
        G = image[:, :, 1]
        B = image[:, :, 0]

        # Obtain luminance using predefined scale factors
        luminance = 0.299 * R + 0.587 * G + 0.114 * B
        mean_luminance = np.mean(luminance)

        # Compute scale factor
        gamma = 2 * (0.5 + mean_luminance)

        # Scale mean-luminance subtracted colour chanels 
        enhanced = np.zeros(image.shape)
        enhanced[:, :, 2] = gamma * (R - mean_luminance)
        enhanced[:, :, 1] = gamma * (G - mean_luminance)
        enhanced[:, :, 0] = gamma * (B - mean_luminance)

        # Clip to [0.0, 1.0]
        enhanced = self.__clip(enhanced)

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
                images=[self.image, enhanced],
                titles=['Original Image', 'Contrast Enhanced Image'],
                size=(15, 15)
            )

        return enhanced

    def luminance_map(self, image=None):
        '''Function to generate the Luminance Weight Map of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)

        # Generate Luminance Map
        luminance = np.mean(image, axis=2)
        luminancemap = np.sqrt((1 / 3) * (np.square(image[:, :, 0] - luminance + np.square(image[:, :, 1] - luminance) + np.square(image[:, :, 2] - luminance))))

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
                images=[self.image, luminancemap],
                titles=['Original Image', 'Luminanace Weight Map'],
                size=(15, 15),
                gray=True
            )
        return luminancemap
    
    def chromatic_map(self, image=None):
        '''Function to generate the Chromatic Weight Map of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)
        
        # Convert to HSV colour space
        hsv = rgb2hsv(image)

        # Extract Saturation
        saturation = hsv[:, :, 1]
        max_saturation = 1.0
        sigma = 0.3
        
        # Generate Chromatic Map
        chromaticmap = np.exp(-1 * (((saturation - max_saturation) ** 2) / (2 * (sigma ** 2))))

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
             images=[self.image, chromaticmap],
             titles=['Original Image', 'Chromatic Weight Map'],
             size=(15, 15),
             gray=True
        )
    
        return chromaticmap

    def saliency_map(self, image=None):
        '''Function to generate the Saliency Weight Map of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)
        
        # Convert image to grayscale
        if(len(image.shape) > 2):
            image = rgb2gray(image)
        else:
            image = image
        
        # Apply Gaussian Smoothing
        gaussian = cv.GaussianBlur(image,(5, 5),0) 
        
        # Apply Mean Smoothing
        image_mean = np.mean(image)
        
        # Generate Saliency Map
        saliencymap = np.absolute(gaussian - image_mean)

        # Display result (if verbose)           
        if self.verbose is True:
            self.__show(
                images=[self.image, saliencymap],
                titles=['Original Image', 'Saliency Weight Map'],
                size=(15, 15),
                gray=True
            )
        
        return saliencymap
    
    def image_pyramid(self, image=None, pyramid_type='gaussian', levels=1):
        '''Function to generate the Gaussian/Laplacian pyramid of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)
        
        # Generate Gaussian Pyramid
        current_layer = image
        gaussian = [current_layer]
        for i in range(levels):
            current_layer = cv.pyrDown(current_layer)
            gaussian.append(current_layer)
            
        if pyramid_type == 'gaussian':
            return gaussian
        # Generate Laplacian Pyramid
        elif pyramid_type == 'laplacian':
            current_layer = gaussian[levels-1]
            laplacian = [current_layer]
            for i in range(levels - 1, 0, -1):
                shape = (gaussian[i-1].shape[1], gaussian[i-1].shape[0])
                expand_gaussian = cv.pyrUp(gaussian[i], dstsize=shape)
                current_layer = cv.subtract(gaussian[i-1], expand_gaussian)
                laplacian.append(current_layer)
            laplacian.reverse()
            return laplacian
            
    def fusion(self, inputs=None, weights=None, gaussians=None):
        '''Function to fuse the pyramids together'''
        # Validate parameters
        if inputs is None or weights is None or gaussians is None:
            return None
        
        fused_levels = []

        # Perform Fusion by combining the Laplacian and Gaussian pyramids
        for i in range(len(gaussians[0])):
            if len(inputs[0].shape) > 2:
                for j in range(inputs[0].shape[2]):
                    # Generate Laplacian Pyramids
                    laplacians = [
                        self.image_pyramid(image=inputs[0][:, :, j], pyramid_type='laplacian', levels=len(gaussians[0])),
                        self.image_pyramid(image=inputs[1][:, :, j], pyramid_type='laplacian', levels=len(gaussians[0]))
                    ]
                    
                    # Adjust rows to match
                    row_size = np.min(np.array([
                        laplacians[0][i].shape[0],
                        laplacians[1][i].shape[0],
                        gaussians[0][i].shape[0],
                        gaussians[1][i].shape[0]
                    ]))

                    # Adjust columns to match
                    col_size = np.min(np.array([
                        laplacians[0][i].shape[1],
                        laplacians[1][i].shape[1],
                        gaussians[0][i].shape[1],
                        gaussians[1][i].shape[1]
                    ]))
                    
                    if j == 0:
                        intermediate = np.ones(inputs[0][:row_size, :col_size].shape)
                    # Fusion Step
                    intermediate[:, :, j] = (laplacians[0][i][:row_size, :col_size] * gaussians[0][i][:row_size, :col_size]) + (laplacians[1][i][:row_size, :col_size] * gaussians[1][i][:row_size, :col_size])
            fused_levels.append(intermediate)
        
        # Reconstruct Image Pyramids
        for i in range(len(fused_levels)-2, -1, -1):
            level_1 = cv.pyrUp(fused_levels[i+1])
            level_2 = fused_levels[i]
            r = min(level_1.shape[0], level_2.shape[0])
            c = min(level_1.shape[1], level_2.shape[1])
            fused_levels[i] = level_1[:r, :c] + level_2[:r, :c]

        # Clip fused image to [0.0, 1.0]
        fused = self.__clip(fused_levels[0])
        if self.verbose is True:
            self.__show(
                    images=[self.image, fused],
                    titles=['Original Image', 'Fusion'],
                    size=(15, 15),
                    gray=False
                )
        return fused

    def dehaze(self, image=None, verbose=None, pyramid_height=12):
        '''Driver function to dehaze the image'''
        # Validate parameters

        # Generating Input Images 
        white_balanced = self.white_balance(image[0])       # First Input Image
        contrast_enhanced = image[1] # Second Input Image
        
        input_images = [
            img_as_float64(white_balanced),
            img_as_float64(contrast_enhanced)
        ]
        
        # Generating Weight Maps
        weight_maps = [
            # Weight maps for first image
            {
                'luminance': self.luminance_map(image=input_images[0]),
                'chromatic': self.chromatic_map(image=input_images[0]),
                'saliency': self.saliency_map(image=input_images[0])
            },
            
            # Weight maps for second image
            {
                'luminance': self.luminance_map(image=input_images[1]),
                'chromatic': self.chromatic_map(image=input_images[1]),
                'saliency': self.saliency_map(image=input_images[1])
            }
        ]
        
        # Weight map normalization
        # Combined weight maps
        weight_maps[0]['combined'] = (weight_maps[0]['luminance'] * weight_maps[0]['chromatic'] * weight_maps[0]['saliency'])
        weight_maps[1]['combined'] = (weight_maps[1]['luminance'] * weight_maps[1]['chromatic'] * weight_maps[1]['saliency'])
        
        # Normalized weight maps
        weight_maps[0]['normalized'] = weight_maps[0]['combined'] / (weight_maps[0]['combined'] + weight_maps[1]['combined'])
        weight_maps[1]['normalized'] = weight_maps[1]['combined'] / (weight_maps[0]['combined'] + weight_maps[1]['combined'])
        
        # Generating Gaussian Image Pyramids
        gaussians = [
            self.image_pyramid(image=weight_maps[0]['normalized'], pyramid_type='gaussian', levels=pyramid_height),
            self.image_pyramid(image=weight_maps[1]['normalized'], pyramid_type='gaussian', levels=pyramid_height)
        ]

        # Fusion Step
        fused = self.fusion(input_images, weight_maps, gaussians)
 
        # Dehazing data
        dehazing = {
            'hazed': self.image,
            'inputs': input_images,
            'maps': weight_maps,
            'dehazed': fused
        }
        
        self.image = None   # Reset image

        return dehazing
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        from collections import namedtuple
from cv2.ximgproc import guidedFilter
from net import *
from net.losses import StdLoss
from net.vae_model import VAE
from utils.imresize import np_imresize
from utils.image_io import *
from utils.dcp import get_atmosphere
from skimage.measure import compare_psnr, compare_ssim
import torch
import torch.nn as nn
import numpy as np

DehazeResult_psnr = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])
DehazeResult_ssim = namedtuple("DehazeResult", ['learned', 't', 'a', 'ssim'])


class Dehaze(object):
    def __init__(self, image_name, image, gt_img, num_iter=500, clip=True, output_path="output/"):
        self.image_name = image_name
        self.image = image
        self.gt_img = gt_img
        self.num_iter = num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None
        self.output_path = output_path

        self.clip = clip
        self.blur_loss = None
        self.best_result_psnr = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 3
        self.post = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.images_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.image_net = image_net.type(data_type)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)

    def _init_ambient(self):
        ambient_net = VAE(self.gt_img.shape)
        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)

        atmosphere = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)
        self.at_back = atmosphere

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        self.image_net_inputs = np_to_torch(self.image).cuda()
        self.mask_net_inputs = np_to_torch(self.image).cuda()
        self.ambient_net_input = np_to_torch(self.image).cuda()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure()
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self):
        self.image_out = self.image_net(self.image_net_inputs)
        self.ambient_out = self.ambient_net(self.ambient_net_input)

        self.mask_out = self.mask_net(self.mask_net_inputs)

        self.blur_out = self.blur_loss(self.mask_out)
        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                     self.images_torch)

        vae_loss = self.ambient_net.getLoss()
        self.total_loss = self.mseloss + vae_loss
        self.total_loss += 0.005 * self.blur_out

        dcp_prior = torch.min(self.image_out.permute(0, 2, 3, 1), 3)[0]
        self.dcp_loss = self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.05
        self.total_loss += self.dcp_loss

        self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        if step % 5 == 0:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
            mask_out_np = self.t_matting(mask_out_np)

            post = np.clip((self.image - ((1 - mask_out_np) * ambient_out_np)) / mask_out_np, 0, 1)

            psnr = compare_psnr(self.gt_img, post)
            ssims = compare_ssim(self.gt_img.transpose(1, 2, 0), post.transpose(1, 2, 0), multichannel=True)

            self.current_result = DehazeResult_psnr(learned=image_out_np, t=mask_out_np, a=ambient_out_np, psnr=psnr)
            self.temp = DehazeResult_ssim(learned=image_out_np, t=mask_out_np, a=ambient_out_np, ssim=ssims)

            if self.best_result_psnr is None or self.best_result_psnr.psnr < self.current_result.psnr:
                self.best_result_psnr = self.current_result

            if self.best_result_ssim is None or self.best_result_ssim.ssim < self.temp.ssim:
                self.best_result_ssim = self.temp

    def _plot_closure(self, step):
        """
         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f  %f cur_ssim %f max_ssim: %f cur_psnr %f max_psnr %f\n' % (
                                                                            step, self.total_loss.item(),
                                                                            self.blur_out.item(),
                                                                            self.temp.ssim,
                                                                            self.best_result_ssim.ssim,
                                                                            self.current_result.psnr,
                                                                            self.best_result_psnr.psnr), '\r', end='')

    def finalize(self):
        self.final_image = np_imresize(self.best_result_psnr.learned, output_shape=self.original_image.shape[1:])
        self.final_t_map = np_imresize(self.best_result_psnr.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result_psnr.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.final_t_map
        self.post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        save_image(self.image_name + "_psnr", self.post, self.output_path)

        self.final_t_map = np_imresize(self.best_result_ssim.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result_ssim.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.final_t_map
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)

        save_image(self.image_name + "_ssim", post, self.output_path)

        self.final_t_map = np_imresize(self.current_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.current_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)

        save_image(self.image_name + "_run_final", post, self.output_path)

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(image_name, image, gt_img, num_iter=500, output_path="output/"):
    dh = Dehaze(image_name, image, gt_img, num_iter, clip=True, output_path=output_path)

    dh.optimize()
    dh.finalize()

    save_image(image_name + "_original", np.clip(image, 0, 1), dh.output_path)

    psnr = dh.best_result_psnr.psnr
    ssim = dh.best_result_ssim.ssim
    return psnr, ssim


if __name__ == "__main__":
    torch.cuda.set_device(0)

    hazy_add = 'data/hazy.png'
    gt_add = 'data/gt.png'
    name = "1400_3"
    print(name)

    hazy_img = prepare_hazy_image(hazy_add)
    gt_img = prepare_gt_img(gt_add, SOTS=True)

    psnr, ssim = dehaze(name, hazy_img, gt_img, output_path="output/")
    print(psnr, ssim)



from collections import namedtuple
from cv2.ximgproc import guidedFilter
from net import *
from net.losses import StdLoss
from net.vae_model import VAE
from utils.imresize import np_imresize
from utils.image_io import *
from utils.dcp import get_atmosphere
from fusion import ImageDehazing
from skimage.io import imread, imsave
#from skimage.measure import compare_psnr, compare_ssim
import torch
import torch.nn as nn
import numpy as np
from newmodel import GridDehazeNet
DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a'])
def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return gradient_h, gradient_y

class Dehaze(object):
    def __init__(self, image_name, image, num_iter=1000, clip=True, output_path="output/"):
        self.image_name = image_name
        self.image = image
        self.num_iter = num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.002
        self.parameters = None
        self.current_result = None
        self.output_path = output_path

        self.clip = clip
        self.blur_loss = None
        self.best_result_psnr = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 3
        self.post = None
        self.adaptiveavg= nn.AdaptiveAvgPool2d(1)
        self.maxpooling=torch.nn.MaxPool2d(7,1)
        self.batch_size = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.images_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        
        self.model  = GridDehazeNet(height=3, width=6, num_dense_layer=4, growth_rate=16).type(data_type)
        #self.image_net=self.image_net.type(torch.cuda.FloatTensor)
        #self.mask_net=self.mask_net.type(torch.cuda.FloatTensor)
        #self.ambient_net=self.ambient_net.type(torch.cuda.FloatTensor)
    
        

    def _init_ambient(self):
        ambient_net = VAE(self.image.shape)
        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)

        atmosphere = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)
        self.at_back = atmosphere

    def _init_parameters(self):
        parameters = [p for p in self.model.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        self.image_net_inputs = np_to_torch(self.image).cuda()


    def _init_all(self):
        self._init_images()
        self._init_nets()
        
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        self.batch_size = self.images_torch.shape
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure()
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self):
        self.image_out, self.mask_out, self.ambient_out = self.model(self.image_net_inputs)

        #self.mask_out[self.mask_out<0.05]=0.05
        self.mask_out=torch.clamp(self.mask_out,0.05,1)
        self.image_out=torch.clamp(self.image_out,0.00,1)
        self.blur_out = self.blur_loss(self.mask_out)
        gradie_h_gt, gradie_v_gt=gradient(self.image_net_inputs)
        
        gradie_h_est, gradie_v_est=gradient(self.image_out)
        
        self.gradie_h_gt=torch.max(gradie_h_est,gradie_h_gt).detach()
        self.gradie_v_gt=torch.max(gradie_v_est,gradie_v_gt).detach()
        L_tran_v = self.mse_loss(gradie_v_est, self.gradie_v_gt)+self.mse_loss(gradie_h_est, self.gradie_h_gt)

        #self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * tmpair,
                                     #self.images_torch)+
        tmpair=self.adaptiveavg((1 - self.mask_out)*self.images_torch)/self.adaptiveavg(1 - self.mask_out)-0.1
        tmpair=torch.clamp(tmpair,0.6,1)

        tmpimg=(self.images_torch-tmpair)/self.mask_out+tmpair
        tmpimg=torch.clamp(tmpimg,0.0,1)
        smooth_x,smooth_y=gradient(self.mask_out)

            
        #print (batch_size)
        #tmpambient_out=tmpair.repeat(1,1,self.batch_size[2],self.batch_size[3])
        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                     self.images_torch)+(torch.mean(smooth_x)+torch.mean(smooth_x))*0.05+L_tran_v*0.5+(torch.mean(gradie_h_est)+torch.mean(gradie_v_est))+self.mse_loss(self.image_out,tmpimg)
        
        self.total_loss = self.mseloss 
        self.total_loss += 0.005 * self.blur_out
        self.total_loss +=self.mse_loss(torch.mean(self.image_out),torch.mean(self.image_net_inputs))
        dcp_prior = torch.max(self.image_out.permute(0, 2, 3, 1), 3)[0]-torch.min(self.image_out.permute(0, 2, 3, 1), 3)[0]

        
        self.dcp_loss = self.mse_loss(dcp_prior, torch.ones_like(dcp_prior)) - 0.05
        self.total_loss += self.dcp_loss
        if (self.current_result):
            self.total_loss += self.mse_loss(self.ambient_out, np_to_torch(self.current_result.a).type(torch.cuda.FloatTensor))
            self.total_loss += self.mse_loss(self.mask_out, np_to_torch(self.current_result.t).type(torch.cuda.FloatTensor))
            self.total_loss += self.mse_loss(self.image_out, np_to_torch(self.current_result.learned).type(torch.cuda.FloatTensor))
        #self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        self.total_loss += self.mse_loss(self.ambient_out, tmpair * torch.ones_like(self.ambient_out))

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        if step % 5 == 0:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
            ambient_out_np[0,:,:]=self.A_matting(ambient_out_np[0,:,:])
            ambient_out_np[1,:,:]=self.A_matting(ambient_out_np[1,:,:])
            ambient_out_np[2,:,:]=self.A_matting(ambient_out_np[2,:,:])
            mask_out_np[mask_out_np<0.1]=0.1
            mask_out_np = self.t_matting(mask_out_np)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np)

    def _plot_closure(self, step):
        """
         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f  %f\n' % (step, self.total_loss.item(), self.blur_out.item()), '\r', end='')

    def finalize(self):
        self.final_t_map = np_imresize(self.current_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.current_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        mask_out_np[mask_out_np<0.1]=0.1
        self.final_a[0,:,:]=self.A_matting(self.final_a[0,:,:])
        self.final_a[1,:,:]=self.A_matting(self.final_a[1,:,:])
        self.final_a[2,:,:]=self.A_matting(self.final_a[2,:,:])
        self.final_a[self.final_a<0.6]=0.6
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        dehazer = ImageDehazing(verbose=False)
        #print (self.original_image.shape,post.shape)
       
        posttest = dehazer.dehaze([np_to_pil(self.original_image),np_to_pil(post)], pyramid_height=15)
        posttest['dehazed'] = np.clip(posttest['dehazed'], 0, 1)
        #posttest['dehazed'].save(output_path + "{}.jpg".format(self.image_name))
        imsave('output/'+self.image_name+'.jpg', posttest['dehazed'])
        save_image(self.image_name + "_run_finala", self.current_result.a, self.output_path)
        save_image(self.image_name + "_run_final_outt", self.current_result.t, self.output_path)

    def A_matting(self, mask_out_np):
        #print (self.original_image.transpose(1, 2, 0).shape,mask_out_np.astype(np.float32).shape)
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np.astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])
    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(image_name, image, num_iter=350, output_path="output/"):
    dh = Dehaze(image_name, image, num_iter, clip=True, output_path=output_path)

    dh.optimize()
    dh.finalize()

    save_image(image_name + "_original", np.clip(image, 0, 1), dh.output_path)

if __name__ == "__main__":
    torch.cuda.set_device(0)

    hazy_add = 'data/canon_input.png'
    name = "canon_input111"
    print(name)

    hazy_img = prepare_hazy_image(hazy_add)
    #print (hazy_img.shape)
    dehaze(name, hazy_img, num_iter=500, output_path="output/")
