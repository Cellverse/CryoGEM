import numpy as np
import torch, torchvision
from cryogem.third_party.kmeans_pytorch import kmeans
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def cosine_window(shape, cutoff_ratio, edge_width=0.01):
    """
    Generate a 2D cosine window filter.
    :param shape: Tuple, the shape of the filter.
    :param cutoff_ratio: float, the cutoff ratio.
    :param edge_width: float, width of the cosine edge relative to the shape's dimensions.
    :return: 2D array, the cosine window filter.
    """
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    distance_from_center = np.sqrt((rows - shape[0] // 2) ** 2 + (cols - shape[1] // 2) ** 2)
    max_distance = np.sqrt((shape[0] // 2) ** 2 + (shape[1] // 2) ** 2)

    cutoff_distance  = cutoff_ratio * max_distance
    transition_width = edge_width * max_distance

    # Create a cosine edge outside the cutoff_distance
    mask = np.where(
        distance_from_center > cutoff_distance + transition_width,
        0,
        np.where(
            distance_from_center < cutoff_distance,
            1,
            0.5 + 0.5 * np.cos(
                np.pi * (distance_from_center - cutoff_distance) / transition_width
            )
        )
    )

    return mask

def histogram_equalization(input_tensor, bins=256, min=0, max=255):
    """ Histogram equalization of a batch of images (float tensors).
    Args:
        input_tensor (torch.Tensor) : input tensor, [B, C, H, W], where C=1
    Returns:
        output_tensor (torch.Tensor): output tensor, [B, C, H, W], where C=1
    
    """

    B, C, H, W = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)

    for b in range(B):
        for channel in range(C):
            # Compute the histogram
            histogram = torch.histc(input_tensor[b, channel] *  max, bins=bins, min=min, max=max)

            # Compute the cumulative distribution function
            cdf = torch.cumsum(histogram, dim=0)
            cdf = cdf / cdf[-1]  # Normalize

            # Approximate interpolation
            input_values = input_tensor[b, channel].flatten() * max
            input_indices = (input_values.long()).clamp(min, max)  # Convert values to indices
            weights = input_values - input_indices.float()       # For linear interpolation

            lower_values = cdf[input_indices]
            upper_values = cdf[(input_indices + 1).clamp(min, max)]
            interp_values = (1 - weights) * lower_values + weights * upper_values

            # Reshape the equalized data to the original shape
            output_tensor[b, channel] = (interp_values.view(H, W) * max).clamp(min, max) / max

    return output_tensor
    
def filter_image(image, cutoff_ratio=0.0425):
    """ Apply a 2D cosine window lowpass filter to the given image.
    Args:
        image: 2D array, the image to be filtered.
        cutoff_ratio: float, the cutoff ratio.
    Returns:
        filtered_image: 2D array, the filtered image.
    """

    assert image.ndim == 2, 'image should be 2D'

    device = image.device

    image_fft = torch.fft.fft2(image)
    cos_window = cosine_window(image.shape, cutoff_ratio)
    cos_window = torch.from_numpy(cos_window).float().to(device)
    filtered_fft = image_fft * torch.fft.fftshift(cos_window) # Apply the filter

    filtered_image = torch.abs(torch.fft.ifft2(filtered_fft)) # Convert back to image
    filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min() + 1e-8)
    filtered_image = histogram_equalization(filtered_image[None, None, ...])                            # float: 0-1

    return filtered_image[0, 0]

# def gaussian_blur(data, kernel_size):
#     """ Apply a gaussian blur to the given data.
#     Args:
#         data: 2D array, the data to be blurred.
#         kernel_size: int, the size of the kernel.
#     """
#     assert data.ndim == 2, 'data should be 2D'
#     blurred_data = torchvision.transforms.GaussianBlur(kernel_size)(data[None, ...])

#     return blurred_data[0]

def window(img, x_patches, y_patches):
    """ Divide the image into patches and compute the mean of each patch.
    Args:
        img: 2D array, the image to be divided into patches.
        x_patches: int, the number of patches in the x direction.
        y_patches: int, the number of patches in the y direction.
    Returns:
        rolled_img: 2D array, the same size as the input image, with the mean of each patch.
    """
    rows, cols = img.shape
    x_patch_size = cols // x_patches
    y_patch_size = rows // y_patches

    # Calculate padding needed
    pad_x = 0 if cols % x_patches == 0 else x_patches - (cols % x_patches)
    pad_y = 0 if rows % y_patches == 0 else y_patches - (rows % y_patches)

    padded_img = img[None, ...]
    padded_img = torch.nn.functional.pad(padded_img, (0, pad_x, 0, pad_y), mode='reflect')
    padded_img = padded_img[0]

    rows, cols = padded_img.shape

    x_patch_size = cols // x_patches
    y_patch_size = rows // y_patches

    assert x_patch_size * x_patches == cols and y_patch_size * y_patches == rows, "Something went wrong!"

    # Reshape the image to get the patches and compute their means
    reshaped = padded_img.reshape(y_patches, y_patch_size, x_patches, x_patch_size)
    patch_means = reshaped.mean(dim=(1, 3))

    # Expand dimensions of the computed means to match the padded image shape
    expanded_means = patch_means.unsqueeze(1).unsqueeze(3)
    expanded_means = expanded_means.expand(-1, y_patch_size, -1, x_patch_size)

    # Reshape back to padded image shape
    mean_map_padded = expanded_means.reshape(rows, cols)

    # Remove the padding to get back the original image shape
    rows, cols = img.shape
    mean_map = mean_map_padded[:rows, :cols]

    return mean_map

def custom_resize(data, downsample_ratio=None, target_size=None, interpolation='bilinear'):
    """ Downsample the given data by the given ratio.
    Args:
        data: 2D array, the data to be downsampled.
        downsample_ratio: int, the ratio by which to downsample the data.
    """
    assert data.ndim == 2, 'data should be 2D'

    if downsample_ratio == None and target_size == None:
        raise ValueError('Either downsample_ratio or target_size should be specified.')
    if downsample_ratio:
        resized_data = torchvision.transforms.Resize(data.shape[0] // downsample_ratio, antialias=True)(data[None, ...])
    if target_size:
        assert isinstance(target_size, list) and len(target_size) == 2, 'target_size should be a list of length 2'
        if interpolation == 'bilinear':
            resized_data = torchvision.transforms.Resize(target_size, antialias=True, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)(data[None, ...])
        elif interpolation == 'nearest':
            resized_data = torchvision.transforms.Resize(target_size, antialias=False, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)(data[None, ...])
        else:
            resized_data = torchvision.transforms.Resize(target_size, antialias=True)(data[None, ...])
    return resized_data[0]  

def custom_KMeans(data, cluster_num):
    """ Apply KMeans clustering to the given 2D image.
    Args:
        data: 2D array, the data to be clustered.
        cluster_num: int, the number of clusters.
    """
    assert data.ndim == 2, 'data should be 2D'
    shape = data.shape
    array = data.reshape(-1, 1)
    label, _ = kmeans(X=array, num_clusters=cluster_num, distance='euclidean', device=data.device, tqdm_flag=False)

    # cluster_res = center[label.flatten()].reshape((shape))
    cluster_res = label.reshape(shape)
    
    return cluster_res

def gaussian_kernel(size: int, sigma: float):
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()  # normalize

    gg = g[:, None] * g[None, :]
    return gg

def gaussian_blur(img, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma).to(img.device)
    
    img = img[None, None, ...].float()
    kernel = kernel[None, None, ...].float()
    
    logger.info(img.shape)
    logger.info(kernel.shape)
    blurred = F.conv2d(img, kernel, padding=kernel_size//2)
    blurred = blurred.cpu().long()

    return blurred[0, 0]

def custom_histogram_equalization_with_kmeans_result(
    data, cluster_res, return_map=False
):
    label = torch.unique(cluster_res)
    label = label.sort()[0]
    result = torch.zeros_like(data)

    for k in label:
        mask = torch.where(cluster_res == k, 1, 0).float().to(data.device)
        mask = custom_resize(mask, target_size=[data.shape[0], data.shape[1]], interpolation='nearest').bool()
        data_k = data[mask]
        if return_map:
            result[mask] += data_k.mean()
        else:
            data_k = histogram_equalization(data_k[None, None, None, ...], bins=256, min=0, max=255)[0, 0, 0]
            result[mask] += data_k
            
    return result

# import time

# if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import imageio.v2 as imageio
    # import mrcfile
#     # Test the function using a sample image
#     # gray_image = plt.imread('sample_image.png')  # Load a sample image
#     # gray_image = imageio.imread('sample_image.png', mode='F')
#     gray_image = imageio.imread('0003.png', mode='F')
#     apix=1.34
#     # apix=0.66
#     gray_image = torch.from_numpy(gray_image).float().cuda()
#     gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-8)
#     # gray_image = custom_resize(gray_image, downsample_ratio=4)
#     # gray_image = gray_image[:512, :512]
#     # apix=apix*4
#     gray_image.requires_grad = True
#     start_time = time.time()
#     lowpass = filter_image(gray_image, cutoff_ratio=0.065*apix)
#     # filtered_image = gaussian_blur(filtered_image, 5)
#     filtered_image = window(lowpass, 20, 20).detach()
#     filtered_image = custom_resize(filtered_image, downsample_ratio=4)
#     cluster_res    = custom_KMeans(filtered_image, 8)
#     result         = custom_histogram_equalization_with_kmeans_result(gray_image, cluster_res)
#     # logger.info(cluster_res.shape)

#     end_time = time.time()
#     logger.info('Time elapsed: {:.2f} seconds'.format(end_time - start_time))
#     logger.info(result.requires_grad)
#     logger.info(result.device)
#     lowpass = lowpass.detach().cpu()
#     cluster_res = cluster_res.detach().cpu()
#     result = result.detach().cpu()
#     gray_image = gray_image.detach().cpu()

#     plt.imsave('result.png', result, cmap='gray')
#     with mrcfile.new('result.mrc', overwrite=True) as mrc:
#         mrc.set_data(result.numpy().astype(np.float32))
#     # Display the results
#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.imshow(gray_image, cmap='gray')
#     plt.title("Original Image")
#     plt.axis('off')

#     plt.subplot(2, 2, 2)
#     plt.imshow(cluster_res, cmap='gray')
#     plt.title("Mask")
#     plt.axis('off')

#     plt.subplot(2, 2, 3)
#     plt.imshow(lowpass, cmap='gray')
#     plt.title("Lowpass")
#     plt.axis('off')

#     plt.subplot(2, 2, 4)
#     plt.imshow(result, cmap='gray')
#     plt.title("Result")
#     plt.axis('off')

#     plt.tight_layout()
#     # plt.show()
#     plt.savefig('sample_image_filtered.png', dpi=300)