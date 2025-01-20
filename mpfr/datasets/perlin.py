import numpy as np

def lerp_np(x,y,w):
    return (1 - w) * x + w * y

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1  

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return np.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

def create_perlin_mask(resize_shape):
    '''
        binary perlin noise mask
    '''
    perlin_scale = 6
    min_perlin_scale = 0
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
    perlin_noise = rand_perlin_2d_np(resize_shape, (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_thr = (perlin_noise > threshold).astype(np.float32)
    perlin_thr = np.expand_dims(perlin_thr, axis=2)
    return perlin_thr 

if __name__ == '__main__':

    import cv2
    size = 512
    perlin_noise = create_perlin_mask([size, size])
    cv2.imwrite('/home/c1c/workbench/base_lab/data/pn_thr_512.png', perlin_noise * 255)
    cv2.imwrite('/home/c1c/workbench/base_lab/data/pn_thr_256.png', cv2.resize(perlin_noise * 255, dsize=(256, 256), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite('/home/c1c/workbench/base_lab/data/pn_thr_64.png', cv2.resize(perlin_noise * 255, dsize=(64, 64), interpolation=cv2.INTER_NEAREST))
    print(perlin_noise.shape)