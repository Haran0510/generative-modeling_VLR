import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision



def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    
    num_samples = 100
    z_dim = 128
    interpolation_steps = 10

    x = torch.linspace(-1, 1, interpolation_steps)
    y = torch.linspace(-1, 1, interpolation_steps)

    x_vector, y_vector = torch.meshgrid(x, y)

    feature_1 = x_vector.flatten().unsqueeze(1)
    feature_2 = y_vector.flatten().unsqueeze(1)
    features_126 = torch.zeros((interpolation_steps**2, z_dim-2))

    z_samples = torch.cat((feature_2, feature_1, features_126), dim=1).to(gen.dense.weight.device)

    generated_samples = gen.forward_given_samples(z_samples)

    torchvision.utils.save_image(
        generated_samples,
        path,
        nrow=interpolation_steps,
        normalize=True,
        range=(-1, 1),
        padding=0,
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
