import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_imgs(images, j=230):
    if len(images) > int(str(j)[0])*int(str(j)[1]):
        return
    for i in range(len(images)):
        j += 1
        plt.subplot(j), plt.imshow(images[i], cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.show()


def upsample(src, big_shape):
    dst = np.zeros(big_shape)
    dst[0::2, 0::2] = src
    return dst


def apply_filter_in_fft(img, fltr):
    img = np.fft.fftshift(np.fft.fft2(img))
    res = img * fltr
    res = np.fft.ifft2(np.fft.ifftshift(res))
    res = np.abs(res)
    return res


def create_gaussian_kernel(shape0, shape1):
    sigmax = (shape0-1)//3
    sigmay = (shape0-1)//3
    rows, cols = shape0, shape1
    cy, cx = rows // 2, cols // 2
    y = np.linspace(0, rows, rows)
    x = np.linspace(0, cols, cols)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-(((x - cx) / sigmax) ** 2 + ((y - cy) / sigmay) ** 2))
    gaussian = (gaussian - np.min(gaussian)) / np.ptp(gaussian).astype('float64')
    return gaussian


def pyr_img(g_a, g_b, g_mask, num_levels):
    g_a = (g_a - np.min(g_a)) / np.ptp(g_a).astype('float64')
    gp_a = [g_a]
    g_b = (g_b - np.min(g_b)) / np.ptp(g_b).astype('float64')
    gp_b = [g_b]
    gp_mask = [g_mask]
    for i in range(num_levels):
        gaussian = create_gaussian_kernel(g_a.shape[0], g_a.shape[1])
        g_a = apply_filter_in_fft(g_a, gaussian)
        g_a = (g_a - np.min(g_a)) / np.ptp(g_a).astype('float64')
        g_a = g_a[0::2, 0::2]
        gp_a.append(np.float64(g_a))
        g_b = apply_filter_in_fft(g_b, gaussian)
        g_b = (g_b - np.min(g_b)) / np.ptp(g_b).astype('float64')
        g_b = g_b[0::2, 0::2]
        gp_b.append(np.float64(g_b))
        g_mask = apply_filter_in_fft(g_mask, gaussian)
        g_mask = (g_mask - np.min(g_mask)) / np.ptp(g_mask).astype('float64')
        g_mask = g_mask[0::2, 0::2]
        gp_mask.append(np.float64(g_mask))
    # plot_imgs(gp_a)
    # plot_imgs(gp_b)
    # plot_imgs(gp_mask)

    lp_a = [gp_a[num_levels - 1]]
    lp_b = [gp_b[num_levels - 1]]
    gp_mask_reverse = [gp_mask[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        upsampled_a = upsample(gp_a[i], gp_a[i - 1].shape)
        gaussian = create_gaussian_kernel(upsampled_a.shape[0], upsampled_a.shape[1])
        upsampled_a = apply_filter_in_fft(upsampled_a, gaussian)
        upsampled_a = (upsampled_a - np.min(upsampled_a)) / np.ptp(upsampled_a).astype('float64')
        l_a = np.subtract(gp_a[i - 1], upsampled_a)

        upsampled_b = upsample(gp_b[i], gp_b[i - 1].shape)
        upsampled_b = apply_filter_in_fft(upsampled_b, gaussian)
        upsampled_b = (upsampled_b - np.min(upsampled_b)) / np.ptp(upsampled_b).astype('float64')
        l_b = np.subtract(gp_b[i - 1], upsampled_b)

        lp_a.append(np.float64(l_a))
        lp_b.append(np.float64(l_b))
        gp_mask_reverse.append(gp_mask[i-1])
    # plot_imgs(lp_a)
    # plot_imgs(lp_b)

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lp_a, lp_b, gp_mask_reverse):
        ls = la * gm + lb * (1.0 - gm)
        ls.dtype = np.float64
        LS.append(ls)
    # plot_imgs(LS)

    # now reconstruct
    ls_ = LS[0]
    ls_.dtype = np.float64
    # toprnt = [ls_]
    for i in range(1, num_levels):
        ls_ = upsample(ls_, LS[i].shape)
        gaussian = create_gaussian_kernel(ls_.shape[0], ls_.shape[1])
        ls_ = apply_filter_in_fft(ls_, gaussian)
        ls_ = (ls_ - np.min(ls_)) / np.ptp(ls_).astype('float64')
        ls_ = cv2.add(ls_, LS[i])
        # toprnt.append(np.float64(ls_))
    # plot_imgs(toprnt)
    return ls_


def main():
    size = (480, 480)
    a = cv2.resize(cv2.imread('pic1.png', cv2.IMREAD_GRAYSCALE), size)
    b = cv2.resize(cv2.imread('pic2.png', cv2.IMREAD_GRAYSCALE), size)
    mask = cv2.resize(cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE), size)
    # normalize to [0, 1]
    mask = (mask - np.min(mask)) / np.ptp(mask).astype('float64')

    num_levels = 4
    g_a = a.copy()
    g_b = b.copy()
    g_mask = mask.copy()
    res = pyr_img(g_a, g_b, g_mask, num_levels)

    plot_imgs([g_a, g_b, res], 130)


if __name__ == "__main__":
    main()
