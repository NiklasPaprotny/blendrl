import numpy as np
import torch as th
from skimage.transform import resize
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter
import cv2


searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def run_through_model(env, model, history, ix, interp_func=None, mask=None, mode='actor'):
    if mask is None:
        im = history['obs'][ix] # (4, 84, 84)
    else:
        assert(interp_func is not None, "interp func cannot be none")
        mask_expanded = np.expand_dims(mask, axis=0) # (1, 84, 84)
        mask_expanded = np.repeat(mask_expanded, 4, axis=0) # Mask reshaped to (4, 84, 84)
        im = interp_func(history['obs'][ix], mask_expanded).reshape(4,84,84)

    state_objects = env.env.objects
    logic_state, neural_state = env.convert_state(state_objects, im) # im has shape (4,84,84), from step() in env.

    neural_state = neural_state.to(model.device)

    actor_value = model.actor.to_neural_action_distribution(neural_state)
    critic_value = model.get_neural_value(neural_state)

    return critic_value if mode == 'critic' else actor_value

def score_frame(env, model, history, ix, r, d, interp_func, mode='actor'):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    L = run_through_model(env, model, history, ix, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(84/d)+1,int(84/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,84,d):
        for j in range(0,84,d):
            mask = get_mask(center=[i,j], size=[84,84], r=r)
            l = run_through_model(env, model, history, ix, interp_func, mask=mask, mode=mode)
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).item()
    pmax = scores.max()
    scores = cv2.resize(scores, (84,84), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return pmax * scores / scores.max()

def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = cv2.resize(saliency, (atari.shape[1], atari.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    I = atari.astype('uint16')
    I[:,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

def get_env_meta(env_name):
    meta = {}
    if env_name=="Pong-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 500
    elif env_name=="Breakout-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 300
    elif env_name=="SpaceInvaders-v0":
        meta['critic_ff'] = 400 ; meta['actor_ff'] = 400
    else:
        print('environment "{}" not supported'.format(env_name))
    return meta